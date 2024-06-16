import numpy as np

from app.game.backened.rule import Rule, CONTINUE, BLACK


class Board(object):
    """棋盘游戏逻辑控制"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))  # 棋盘宽度
        self.height = int(kwargs.get('height', 15))  # 棋盘高度
        self.states = {}  # 棋盘状态为一个字典,键: 移动步数,值: 玩家的棋子类型
        self.n_in_row = int(kwargs.get('n_in_row', 5))  # 5个棋子一条线则获胜
        self.players = [1, 2]  # 玩家1,2
        self.rule = Rule()

    def init_board(self, start_player=0):
        # 初始化棋盘
        # 当前棋盘的宽高小于5时,抛出异常(因为是五子棋)
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('棋盘的长宽不能少于{}'.format(self.n_in_row))
        self.start_player = start_player
        self.current_player = self.players[start_player]  # 先手玩家
        self.availables = list(range(self.width * self.height))  # 初始化可用的位置列表
        self.states = {}  # 初始化棋盘状态
        self.last_move = -1  # 初始化最后一次的移动位置

        self.rule = Rule()  # 注意规则
        return self

    def move_to_location(self, move):
        # 根据传入的移动步数返回位置(如:move=2,计算得到坐标为[0,2],即表示在棋盘上左上角横向第三格位置)
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        # 根据传入的位置返回移动值
        # 位置信息必须包含2个值[h,w]
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        # 超出棋盘的值不存在
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        从当前玩家的角度返回棋盘状态。
        状态形式：3 * 宽 * 高
        """
        # 使用3个15x15的二值特征平面来描述当前的局面
        # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0
        # 第三个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为-1，否则全部为1

        square_state = np.zeros((3, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]  # 获取棋盘状态上属于当前玩家的所有移动值
            move_oppo = moves[players != self.current_player]  # 获取棋盘状态上属于对方玩家的所有移动值
            square_state[0][move_curr // self.width,  # 对第一个特征平面填充值(当前玩家)
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,  # 对第二个特征平面填充值(对方玩家)
                            move_oppo % self.height] = 1.0
        # 指出当前玩家的颜色
        square_state[2][:, :] = -1.0 if self.current_player == self.players[self.start_player] else 1.0
        # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行,第二行换到倒数第二行..)
        # return square_state[:, ::-1, :].copy()
        return square_state.copy()

    #     def current_state(self):
    #         """
    #         从当前玩家的角度返回棋盘状态。
    #     状态形式：4 * 宽 * 高
    #         """
    #         # 使用4个15x15的二值特征平面来描述当前的局面
    #         # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0
    #         # 第三个平面表示对手player最近一步的落子位置，也就是整个平面只有一个位置是1，其余全部是0
    #         # 第四个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
    #         square_state = np.zeros((4, self.width, self.height))
    #         if self.states:
    #             moves, players = np.array(list(zip(*self.states.items())))
    #             move_curr = moves[players == self.current_player]  # 获取棋盘状态上属于当前玩家的所有移动值
    #             move_oppo = moves[players != self.current_player]  # 获取棋盘状态上属于对方玩家的所有移动值
    #             square_state[0][move_curr // self.width,  # 对第一个特征平面填充值(当前玩家)
    #                             move_curr % self.height] = 1.0
    #             square_state[1][move_oppo // self.width,  # 对第二个特征平面填充值(对方玩家)
    #                             move_oppo % self.height] = 1.0
    #             # 指出最后一个移动位置
    #             square_state[2][self.last_move // self.width,  # 对第三个特征平面填充值(对手最近一次的落子位置)
    #                             self.last_move % self.height] = 1.0
    #         if len(self.states) % 2 == 0:  # 对第四个特征平面填充值,当前玩家是先手,则填充全1,否则为全0
    #             square_state[3][:, :] = 1.0
    #         # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行,第二行换到倒数第二行..)
    #         return square_state[:, ::-1, :].copy()

    def do_move(self, move):
        # 根据移动的数据更新各参数
        self.states[move] = self.current_player  # 将当前的参数存入棋盘状态中
        self.availables.remove(move)  # 从可用的棋盘列表移除当前移动的位置
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )  # 改变当前玩家
        self.last_move = move  # 记录最后一次的移动位置

        self.rule.play(move)
        self.availables = self.rule.get_available()

    def has_a_winner(self):
        winner = self.rule.game_over_who_win()
        if winner == CONTINUE:
            # 当前都没有赢家,返回False
            return False, -1
        elif winner == BLACK:  # 玩家1
            return True, self.players[0]
        else:
            return True, self.players[1]

        # # 是否产生赢家
        # width = self.width  # 棋盘宽度
        # height = self.height  # 棋盘高度
        # states = self.states  # 状态
        # n = self.n_in_row  # 获胜需要的棋子数量
        #
        # # 当前棋盘上所有的落子位置
        # moved = list(set(range(width * height)) - set(self.availables))
        # if len(moved) < self.n_in_row + 2:
        #     # 当前棋盘落子数在7个以上时会产生赢家,落子数低于7个时,直接返回没有赢家
        #     return False, -1
        #
        # # 遍历落子数
        # for m in moved:
        #     h = m // width
        #     w = m % width  # 获得棋子的坐标
        #     player = states[m]  # 根据移动的点确认玩家
        #
        #     # 判断各种赢棋的情况
        #     # 横向5个
        #     if (w in range(width - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
        #         return True, player
        #
        #     # 纵向5个
        #     if (h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
        #         return True, player
        #
        #     # 左上到右下斜向5个
        #     if (w in range(width - n + 1) and h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
        #         return True, player
        #
        #     # 右上到左下斜向5个
        #     if (w in range(n - 1, width) and h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
        #         return True, player
        #
        # # 当前都没有赢家,返回False
        # return False, -1

    def game_end(self):
        """检查当前棋局是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            # 棋局布满,没有赢家
            return True, -1
        return False, -1

    def visual(self):
        print('*' * 50)
        for y in reversed(range(self.height)):
            print(f"{y}:\t", end='')
            for x in range(self.width):
                move = self.location_to_move((x, y))
                if move in self.states.keys():

                    if self.states[move] == self.players[0]:
                        if move == self.last_move:
                            print('\033[0;33;40m1\033[0m\t', end='')
                        else:
                            print('\033[0;31;40m1\033[0m\t', end='')
                    elif self.states[move] == self.players[1]:
                        if move == self.last_move:
                            print('\033[0;33;40m2\033[0m\t', end='')
                        else:
                            print('\033[0;32;40m2\033[0m\t', end='')
                        # print(f'{self.xy2move(x,y)}',end=' ')
                else:
                    print('-\t', end='')
                    # print(f'{self.xy2move(x,y)}',end=' ')
            print()
        print(' \t', end='')
        for x in range(self.width):
            print(f"{x}\t", end='')
        print()
        print('*' * 50)

    def get_current_player(self):
        return self.current_player


class Game():
    def __init__(self, board: Board, is_shown=False):
        self.board = board
        self.is_shown = is_shown

    def start_self_play(self, player, temp=1e-3):
        """
        使用MCTS玩家开始自己玩游戏,重新使用搜索树并存储自己玩游戏的数据
        (state, mcts_probs, z) 提供训练
        :param player:
        :param temp:
        :return:
        """
        self.board.init_board()  # 初始化棋盘

        # 疏影局训练
        self.board.do_move(112)
        self.board.do_move(113)
        self.board.do_move(9 * 15 + 9)
        # todo

        states, mcts_probs, current_players = [], [], []  # 状态,mcts的行为概率,当前玩家

        if self.is_shown:
            self.board.visual()

        while True:
            # 根据当前棋盘状态返回可能得行为,及行为对应的概率
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # 存储数据
            states.append(self.board.current_state())  # 存储状态数据
            mcts_probs.append(move_probs)  # 存储行为概率数据
            current_players.append(self.board.current_player)  # 存储当前玩家
            # 执行一个移动
            self.board.do_move(move)
            if self.is_shown:
                self.board.visual()

            # 判断该局游戏是否终止
            end, winner = self.board.game_end()
            if end:
                # 从每个状态的当时的玩家的角度看待赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    # 没有赢家时
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MSCT的根节点
                player.reset_player()
                if self.is_shown:
                    print(f"winner : \t{winner}")
                return winner, zip(states, mcts_probs, winners_z)

    def start_play(self, player1, player2, start_player=0):
        """开始一局游戏"""
        if start_player not in (0, 1):
            # 如果玩家不在玩家1,玩家2之间,抛出异常
            raise Exception('开始的玩家必须为0(玩家1)或1(玩家2)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = self.board.players  # 加载玩家1,玩家2
        player1.set_player_ind(p1)  # 设置玩家1
        player2.set_player_ind(p2)  # 设置玩家2
        players = {p1: player1, p2: player2}
        if self.is_shown:
            self.board.visual()
        while True:
            current_player = self.board.current_player  # 获取当前玩家
            player_in_turn = players[current_player]  # 当前玩家的信息
            move = player_in_turn.get_action(self.board)  # 基于MCTS的AI下一步落子
            self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数
            if self.is_shown:
                # 展示棋盘
                self.board.visual()
            # 判断当前棋局是否结束
            end, winner = self.board.game_end()
            # 结束
            if end:
                win = winner
                break
        if self.is_shown:
            print(f"winner : \t{win}")
        return win


if __name__ == "__main__":
    board = Board(width=15, height=15, n_in_row=5)
    board.init_board()
    board.do_move(112)
    board.do_move(113)
    board.do_move(9 * 15 + 9)
    board.visual()
    # game = Game(board, True)
    #
    # player1 = MCTS_Pure(c_puct=5.0,n_playout=1000)
    # player2 = MCTS_Pure(c_puct=5.0,n_playout=2000)
    # game.start_play(player1=player1,player2=player2)
