from copy import deepcopy

import torch

# 棋盘上位置的状态
EMPTY = -1
WHITE = 0
BLACK = 1
# 游戏胜利与否的状态
CONTINUE = 2
NO_CLEAR_WINNER = 3  # 平局
# 棋盘大小
BOARD_LEN = 15
# 神经网络输入1*3*15*15的特征
FEATURE_LEN = 3

directions = [[[0, -1], [0, 1]],  # 竖直搜索
              [[-1, 0], [1, 0]],  # 水平搜索
              [[-1, -1], [1, 1]],  # 主对角线搜索
              [[1, -1], [-1, 1]]]  # 副对角线搜索


class Rule:
    """
    connect5的棋盘类
    """

    def __init__(self, board=None, player=BLACK, turn=0):
        """
        棋盘board []  元素范围0-224
        行棋方player  WHITE,BLACK
        历史记录history 元素(第几回合,那方,落子位置0-255)
        回合数turn 已经落了几个子
        """
        if board is None:
            self.board = [EMPTY for i in range(BOARD_LEN * BOARD_LEN)]
        else:
            self.board = board
        self.player = player
        # 元素(第几回合,那方,落子位置0-255)
        self.history = []
        self.turn = turn

    def copy(self):
        return deepcopy(self)

    def play(self, act):
        """
        落子到act
        注意回合数加一的同时下棋方改变了
        """
        cur_player = self.player
        self.board[act] = self.player
        self.history.append((self.turn, self.player, act))
        self.turn += 1
        self.player = BLACK + WHITE - self.player

        # print([pre_act for pre_turn, pre_player, pre_act in self.history])
        # sstr = "{"
        # for pre_turn, pre_player, pre_act in self.history:
        #     sstr += str(pre_act)
        #     if pre_act != self.history[-1][2]:
        #         sstr += ","
        # sstr += "} "
        # sstr += "nums:" + str(len(self.history))
        # print(sstr)

        # # 调用一下连通子图
        #
        # self.pre_act_connect_graph.clear()
        # self.has_all_connect_point(act)

    def un_play(self):
        """
        悔棋
        """
        pre_turn, pre_player, pre_act = self.history.pop()
        self.turn = pre_turn
        self.board[pre_act] = EMPTY
        self.player = pre_player

        # self.pre_act_connect_graph.clear()
        # self.has_all_connect_point(pre_act)

    # def get_feature(self) -> torch.Tensor:
    #     """
    #     返回向神经网络输入的特征 3*15*15
    #     3*15*15
    #     第一张15*15的棋盘是我方有棋子处为1，其余位置都为0
    #     第二张15*15的棋盘是对方有棋子处为1，其余位置都为0
    #     第三张15*15的棋盘是如果我方是黑方，全为-1，我方是白方，全为1
    #     """
    #     feature = torch.zeros((FEATURE_LEN, BOARD_LEN, BOARD_LEN))
    #
    #     for i in range(BOARD_LEN):
    #         for j in range(BOARD_LEN):
    #             pos = i * BOARD_LEN + j
    #             if self.board[pos] == self.player:
    #                 feature[0][i][j] = 1
    #             elif self.board[pos] == WHITE + BLACK - self.player:
    #                 feature[1][i][j] = 1
    #     if self.player == BLACK:
    #         feature[2] = -1
    #     elif self.player == WHITE:
    #         feature[2] = 1
    #
    #     return feature

    def get_available(self) -> list:
        """
        返回当前方在当前棋盘可以行棋的位置的list
        """
        res = []
        for i in range(BOARD_LEN * BOARD_LEN):
            if self.board[i] == EMPTY and self.is_not_ban(i):
                res.append(i)
        return res

    def __str__(self):
        j = BOARD_LEN - 1
        while j >= 0:
            i = 0
            while i < BOARD_LEN:
                if i == 0:
                    print(f"{j+1}\t :",end=" ")
                if self.board[int(i * BOARD_LEN) + j] == EMPTY:
                    print("_ ",end=" ")
                elif self.board[int(i * BOARD_LEN) + j] == BLACK:
                    print("B ",end=" ")
                else:
                    print("W ",end=" ")
                i += 1
            print("\n",end=" ")
            if j == 0:
                print("\t  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O\n",end=" ")
            j -= 1
        print("\n",end=" ")

    def game_over_who_win(self):
        """
        返回胜利方 WHITE BLACK 或则 还未结束 CONTINUE
        """

        if self.turn <= 5:
            return CONTINUE

        # 需注意play方法过后 双方会变
        pre_turn, pre_player, pre_act = self.history[-1]

        # 查看黑棋白棋有没有连5子
        if self.connect_5(one_x=pre_act, color=pre_player):
            return pre_player

        # 如果没有连成5个子， 需要查看是否违规禁手
        if pre_player == BLACK:
            if pre_player == BLACK:
                if not self.is_not_ban(pre_act):
                    return WHITE

        # 查看下是否已经落满棋盘 如果落满 则为平局，否则返回继续continue
        if len(self.history) == 225:
            return NO_CLEAR_WINNER

        return CONTINUE

    def long_connect(self, one_x) -> bool:
        """
        黑棋在one_x各个方向上是否形成长连禁手
        """
        row, col = one_x // BOARD_LEN, one_x % BOARD_LEN,
        for i in range(4):
            number = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < BOARD_LEN and 0 <= col_t < BOARD_LEN and self.board[
                        int(row_t * BOARD_LEN + col_t)] == BLACK:
                        number += 1
                        if number > 5:
                            return True
                    else:
                        flag = False
        return False

    def connect_5(self, one_x, color) -> bool:
        """
        检查one_x各方向是否连成5子
        """
        row, col = one_x // BOARD_LEN, one_x % BOARD_LEN
        for i in range(4):  # 四个方向
            number = 1
            for j in range(2):  # 方向左右
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < BOARD_LEN and 0 <= col_t < BOARD_LEN and self.board[
                        int(row_t * BOARD_LEN + col_t)] == color:
                        number += 1
                    else:
                        flag = False
            if color == BLACK and number == 5:  # 黑棋必须严格等于5
                return True
            if color == WHITE and number >= 5:  # 白棋长连也算赢
                return True
        return False

    def is_not_ban(self, one_x) -> bool:
        """
        检查黑棋在one_x是否不是禁手
        """
        if self.long_connect(one_x):
            # print(f"{one_x // BOARD_LEN},{one_x % BOARD_LEN}，违背了长连禁手")
            return False
        if self.three_ban():
            # print(f"{one_x // BOARD_LEN},{one_x % BOARD_LEN}，违背了三三禁手")
            return False
        if self.four_ban():
            # print(f"{one_x // BOARD_LEN},{one_x % BOARD_LEN}，违背了四四禁手")
            return False

        return True

    def three_ban(self):
        three_count = 0
        ct1 = "EEBBBE"
        ct2 = "EBBBEE"
        jt1 = "EBEBBE"
        jt2 = "EBBEBE"

        pre_turn, pre_player, pre_act = self.history[-1]
        x = pre_act // BOARD_LEN
        y = pre_act % BOARD_LEN
        m_str = ""
        # 水平方向
        start_x = x - min(x, 4)
        end_x = min(x + 4 + 1, BOARD_LEN)
        j = y
        for i in range(start_x, end_x):
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
        pos = m_str.find(ct1)
        if pos != -1:
            if not self.three_special_case(m_str, pos, 1):
                three_count += 1
        else:
            pos = m_str.find(ct2)
            if pos != -1:
                if not self.three_special_case(m_str, pos, 2):
                    three_count += 1
        pos = m_str.find(jt1)
        if pos != -1:
            three_count += 1
        pos = m_str.find(jt2)
        if pos != -1:
            three_count += 1
        if three_count > 1:
            return True
        # 竖直方向
        m_str = ""
        start_y = y - min(y, 4)
        end_y = min(y + 4 + 1, BOARD_LEN)
        i = x
        for j in range(start_y, end_y):
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
        pos = m_str.find(ct1)
        if pos != -1:
            if not self.three_special_case(m_str, pos, 1):
                three_count += 1
        else:
            pos = m_str.find(ct2)
            if pos != -1:
                if not self.three_special_case(m_str, pos, 2):
                    three_count += 1
        pos = m_str.find(jt1)
        if pos != -1:
            three_count += 1
        pos = m_str.find(jt2)
        if pos != -1:
            three_count += 1
        if three_count > 1:
            return True
        # y=x方向
        m_str = ""
        bias = min(min(x, 4), min(y, 4))
        bias_end = min(min(x + 4, BOARD_LEN - 1)-x, min(y + 4, BOARD_LEN - 1)-y)
        start_x = x - bias
        start_y = y - bias
        end_x = x + bias_end + 1
        end_y = y + bias_end + 1
        for (i, j) in zip(range(start_x, end_x), range(start_y, end_y)):
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
        pos = m_str.find(ct1)
        if pos != -1:
            if not self.three_special_case(m_str, pos, 1):
                three_count += 1
        else:
            pos = m_str.find(ct2)
            if pos != -1:
                if not self.three_special_case(m_str, pos, 2):
                    three_count += 1
        pos = m_str.find(jt1)
        if pos != -1:
            three_count += 1
        pos = m_str.find(jt2)
        if pos != -1:
            three_count += 1
        if three_count > 1:
            return True
        # y=-x方向
        m_str = ""
        bias = min(min(x, 4),  min(y + 4, BOARD_LEN - 1)-y)
        bias_end = min(min(y, 4), min(x + 4, BOARD_LEN - 1)-x)
        start_x = x - bias
        start_y = y + bias
        end_x = x + bias_end + 1
        end_y = y - bias_end - 1
        i = start_x
        j = start_y
        while i < end_x and j > end_y:
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
            i += 1
            j -= 1

        pos = m_str.find(ct1)
        if pos != -1:
            if not self.three_special_case(m_str, pos, 1):
                three_count += 1
        else:
            pos = m_str.find(ct2)
            if pos != -1:
                if not self.three_special_case(m_str, pos, 2):
                    three_count += 1
        pos = m_str.find(jt1)
        if pos != -1:
            three_count += 1
        pos = m_str.find(jt2)
        if pos != -1:
            three_count += 1
        if three_count > 1:
            return True
        return False

    def three_special_case(self, m_str, pos, three_case):
        if three_case == 1:
            if pos + 6 < len(m_str):
                if m_str[pos + 6] == 'B':
                    return True
        else:
            if pos > 0:
                if m_str[pos - 1] == 'B':
                    return True
        return False

    def four_special_case(self, m_str, pos, four_case):
        if four_case == 1:
            if pos > 0:
                if m_str[pos - 1] == 'B':
                    return True
            if pos + 5 < len(m_str):
                if m_str[pos + 5] == 'B':
                    return True
            return False
        elif four_case == 2:
            if pos > 0:
                if pos + 6 < len(m_str):
                    if m_str[pos - 1] == 'B' and (
                            (m_str[pos + 5] == 'E' and m_str[pos + 6] == 'B') or m_str[pos + 5] == 'W'):
                        return True
                    return False
                if pos + 5 < len(m_str):
                    if m_str[pos - 1] == 'B' and m_str[pos + 5] == 'W':
                        return True
                    return False
                if m_str[pos - 1] == 'B':
                    return True
                return False
            else:
                return False
        else:
            if pos + 5 < len(m_str):
                if pos - 2 >= 0:
                    if (m_str[pos - 2] == 'B' and m_str[pos - 1] == 'E') or m_str[pos - 1] == 'W' and m_str[
                        pos + 5] == 'B':
                        return True
                    return False
                elif pos - 1 >= 0:
                    if m_str[pos + 5] == 'B' and m_str[pos - 1] == 'W':
                        return True
                    return False
                if m_str[pos + 5] == 'B':
                    return True
                return False
            else:
                return False

    def four_ban(self):
        jf1 = "BBBEB"
        jf2 = "BEBBB"
        jf3 = "BBEBB"
        cf1 = "EBBBB"
        cf2 = "BBBBE"
        pre_turn, pre_player, pre_act = self.history[-1]
        x = pre_act // BOARD_LEN
        y = pre_act % BOARD_LEN
        m_str = ""
        four_count = 0
        # 水平方向
        start_x = x - min(x, 5)
        end_x = min(x + 5 + 1, BOARD_LEN)
        j = y
        for i in range(start_x, end_x):
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
        pos = m_str.find(cf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 2):
                four_count += 1
        else:
            pos = m_str.find(cf2)
            if pos != -1:
                if not self.four_special_case(m_str, pos, 3):
                    four_count += 1
        pos = m_str.find(jf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf2)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf3)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        if four_count > 1:
            return True

            # 竖直方向
        #竖直
        m_str = ""
        start_y = y - min(y, 5)
        end_y = min(y + 5 + 1, BOARD_LEN)
        i = x
        for j in range(start_y, end_y):
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
        pos = m_str.find(cf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 2):
                four_count += 1
        else:
            pos = m_str.find(cf2)
            if pos != -1:
                if not self.four_special_case(m_str, pos, 3):
                    four_count += 1
        pos = m_str.find(jf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf2)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf3)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        if four_count > 1:
            return True
        # y=x方向
        m_str = ""
        bias = min(min(x, 5), min(y, 5))
        bias_end = min(min(x + 5, BOARD_LEN - 1)-x,  min(y + 5,BOARD_LEN - 1)-y)
        start_x = x - bias
        start_y = y - bias
        end_x = x + bias_end + 1
        end_y = y + bias_end + 1
        for (i, j) in zip(range(start_x, end_x), range(start_y, end_y)):
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
        pos = m_str.find(cf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 2):
                four_count += 1
        else:
            pos = m_str.find(cf2)
            if pos != -1:
                if not self.four_special_case(m_str, pos, 3):
                    four_count += 1
        pos = m_str.find(jf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf2)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf3)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        if four_count > 1:
            return True

        # y=-x
        m_str = ""

        bias = min(min(x, 5),  min(y + 5, BOARD_LEN - 1)-y)
        bias_end = min(min(y, 5), min(x + 5, BOARD_LEN - 1)-x)
        start_x = x - bias
        start_y = y + bias
        end_x = x + bias_end + 1
        end_y = y - bias_end - 1
        i = start_x
        j = start_y
        while i < end_x and j > end_y:
            if self.board[i * BOARD_LEN + j] == EMPTY:
                m_str += "E"
            elif self.board[i * BOARD_LEN + j] == BLACK:
                m_str += "B"
            else:
                m_str += "W"
            i += 1
            j -= 1

        pos = m_str.find(cf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 2):
                four_count += 1
        else:
            pos = m_str.find(cf2)
            if pos != -1:
                if not self.four_special_case(m_str, pos, 3):
                    four_count += 1
        pos = m_str.find(jf1)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf2)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        pos = m_str.find(jf3)
        if pos != -1:
            if not self.four_special_case(m_str, pos, 1):
                four_count += 1
        if four_count > 1:
            return True
        return False



