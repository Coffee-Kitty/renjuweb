var address= "http://127.0.0.1:5500/";

class Player {

    pla
    constructor() {
        if (this.constructor === Player) {
            throw new Error("Cannot instantiate abstract class Player");
        }
    }
    async getNextMove(chessboard,x,y) {
        throw new Error("Method 'getNextMove' must be implemented");
    }
}
class HumanPlayer extends Player {

    constructor() {
        super();
    }


    async getNextMove(chessboard,x,y) {

        return await axios.post(address+'pk/getMove', { player:'player1',game_id:human_player.game_id,name:human_player.name,player1_name:human_player.name,player2_name:botPlayer.name,chesshistory: chesshistory,x:x,y:y})
        .then(function (response) {
            //无需ai处理

            console.log(response.data);
            // 获取下一步动作的x和y坐标
            // data=JSON.parse(response.data);
            let data=response.data;
            console.log(data);
            x = data.x;
            y = data.y;
            let has_winner = data.has_winner;
            console.log(typeof(has_winner));
            has_winner = has_winner === 1;
            let winner= data.winner;
            console.log([x,y,has_winner,winner]);
            return Promise.resolve([x,y,has_winner,winner]);
        })
        .catch(function (error) {
            console.error('Error:', error);
            return Promise.resolve([x,y,false,-1]);
        });

    }
}
class BotPlayer extends Player {
    constructor() {
        super();
    }
    async getNextMove(chessboard,x,y) {

        // Send chessboard data to the backend
        return await axios.post(address+'pk/getMove', { player:'player2',name:botPlayer.name,game_id:botPlayer.game_id,player1_name:human_player.name,player2_name:botPlayer.name,chesshistory: chesshistory})
        .then(function (response) {
            console.log(response.data);
            // 获取下一步动作的x和y坐标
            // data=JSON.parse(response.data);
            let data=response.data;
            console.log(data);
            x = data.x;
            y = data.y;
            let has_winner = data.has_winner;
            console.log(typeof(has_winner));
            has_winner = has_winner === 1;
            let winner= data.winner;
            console.log([x,y,has_winner,winner]);
            return Promise.resolve([x,y,has_winner,winner]);
        })
        .catch(function (error) {
            console.error('Error:', error);
            return Promise.resolve([0,0,false,-1]);
        });
    }
}
var human_player=new HumanPlayer();
var botPlayer=new BotPlayer();
var isBlack=true;
function setCookie(name, value, days) {
    let expires = "";
    if (days) {
        let date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "") + expires + "; path=/";
}

function getCookie(name) {
    let nameEQ = name + "=";
    let ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
}

function parseCookies() {
    let cookies = {};
    document.cookie.split(';').forEach(cookie => {
        let [name, value] = cookie.split('=').map(c => c.trim());
        cookies[name] = value;
    });
    return cookies;
}

function clearCookies() {
    let cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
        let cookie = cookies[i];
        let eqPos = cookie.indexOf("=");
        let name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
    }
}

window.onload=function(){
    restart();
    // clearCookies();
    // setCookie('player1','admin');
    // setCookie('player2','coffeecat');
    // setCookie('game_id','0');
   memory_id=getCookie('memory_id');

   chesshistory=new Array(0);

   axios.post(address+'history/getMemory', {id: memory_id})
        .then(function (response) {
            console.log(response.data);
            human_player.name=response.data.history.player1_name;
            botPlayer.name=response.data.history.player2_name;
            human_player.game_id=response.data.history.id;
            botPlayer.game_id=response.data.history.id;
            moves=response.data.moves;
            console.log(moves);
            for(i = 0; i<moves.length; i++){
                domove(moves[i][0],moves[i][1]);
            }



            if(isBlack){
                 turn=BLACK;
            }

            else{
                 turn=WHITE;
            }



            human_player.avatar='image/login.jpg';
            botPlayer.avatar='image/login.jpg';
            if(isBlack){
               document.getElementById('player1-name').textContent = human_player.name+"执黑棋";
               document.getElementById('player2-name').textContent = botPlayer.name+"执白棋";
            }else{
                document.getElementById('player1-name').textContent = human_player.name+"执白棋";
               document.getElementById('player2-name').textContent = botPlayer.name+"执黑棋";
            }


            document.getElementById('player1-avatar').src = human_player.avatar || 'image/login.jpg';


            document.getElementById('player2-avatar').src = botPlayer.avatar || 'image/login.jpg';
        });
    // has_winner=false;


    // clearCookies();
}
var board_width=15,board_height=15;//棋盘宽高
window.EMPTY = -1;//棋子类型
window.BLACK = 0;
window.WHITE = 1;
var chesshistory=new Array(0);//落子history
var chessboard=[];//棋盘
for(var i=0;i<board_width;i++){
    chessboard[i] = [];
    for(var j=0;j<board_height;j++){
        chessboard[i][j]=EMPTY;
    }
}




var turn=BLACK;//当前回合
var isOver=false;//是否获胜


var chess = document.getElementsByClassName("chess")[0];
var context = chess.getContext("2d");

//ui位置
var block_chess_width=chess.width/board_width;
var start_x=block_chess_width/2;
var end_x=block_chess_width*(board_width-1)+start_x;
var block_chess_height=chess.height/board_height;
var start_y=block_chess_height/2;
var end_y=block_chess_height*(board_height-1)+start_y;

// console.log(block_chess_width);
// console.log(chess.width);
// console.log(block_chess_width);
// console.log(chess.height);
//画棋盘
function drawChessBoard(){
    let i;
    context.clearRect(0, 0, chess.width, chess.height); // 清空画布
    context.strokeStyle = "black"; //线条颜色
    context.lineWidth = 0.2;


    for(i = 0; i<board_height; i++){
        //画横线
        context.moveTo(start_x,i*block_chess_height+start_y);
        context.lineTo(end_x,i*block_chess_height+start_y);
        context.stroke();
    }
    for(i = 0; i<board_width; i++){
        //画竖线
        context.moveTo(i*block_chess_width+start_x,start_y);
        context.lineTo(i*block_chess_width+start_x,end_y);
        context.stroke();
    }
}
//画棋子
function drawChess(){
    console.log(chesshistory);
    for(let index=0;index<chesshistory.length;index++){
            let value=chesshistory[index];
            // console.log(value);
            let i=value.x,j=value.y;
            // console.log(i);
            // console.log(j);
            context.beginPath();

            if(chessboard[i][j]!==EMPTY){
                let x=start_x+i*block_chess_width;
                let y=start_y+(board_height-1-j)*block_chess_height;
                var radius=(block_chess_height+block_chess_width)/5;
                context.arc(x,y,radius,0,2*Math.PI);
                if(chessboard[i][j]===BLACK){
                    context.fillStyle="black";


                }else if(chessboard[i][j]===WHITE){
                    context.fillStyle="white";
                }
                context.fill();
                context.fillStyle="red";
                context.fillText(index+1,x,y);

            }
            context.closePath();

    }
}

//胜负
//对局结束
function GameOver(has_winner,winner){
    console.log(has_winner);
    if(has_winner){
        console.log("ok");
        // window.alert("GameOver");
        if(winner===window.BLACK){

            window.confirm("BLACK is win");
            addMessage("BLACK is win");
        }
        else if(winner===window.WHITE){
            window.confirm("WHITE is win");
            addMessage("WHITE is win");
        }else{
            window.confirm("DRAW");
            addMessage("DRAW");
        }
        isOver=true;
    }
}

//重新开始对局
function restart(){
    context.clearRect(0, 0, chess.width, chess.height); // 清空画布
    for(var i=0;i<board_width;i++){
        chessboard[i] = [];
        for(var j=0;j<board_height;j++){
            chessboard[i][j]=EMPTY;
        }

    }


    // domove(0,12);
    // domove(0,1);
    // domove(0,9);
    // domove(0,3);
    // domove(0,14);
    // domove(0,0);
    // domove(0,11);
    // domove(0,4);

    // domove(7,7);
    // domove(7,8);
    // domove(9,9);

    drawChessBoard();
    drawChess();

    initMessage("Gomoku游戏规则：\n" +
        "在简单的五子棋上增加了黑棋的禁手规则\n"+
        "详情请参见http://computergames.caai.cn/jsgz16.html\n"+
        "本站仅作简单演示\n" +
        "不增加三手交换和五手N打规则,"+
        "且只选择一种特定开局");
}
// document.getElementsByClassName("restart")[0].onclick=restart;

// var has_winner=false;
//落子
async function domove(x,y){
    if(x<0||x>board_width||y<0||y>board_height)return;
    // 已经落子同样g掉
    if(chessboard[x][y]!==EMPTY){
        addMessage("不可重复落子");
        return;
    }


    let result=await human_player.getNextMove(chessboard,x,y);

    console.log(result);
    let nextX=result[0];
    let nextY=result[1];
    console.log(nextX);
    console.log(nextY);
    if(chessboard[nextX][nextY]===EMPTY){
        chessboard[nextX][nextY]=turn;
        chesshistory.push({x: nextX, y: nextY, color: turn});
        turn=BLACK+WHITE-turn;
    }
    drawChess();
    // addMessage("落子"+"("+x+","+y+")");

    has_winner=result[2];
    console.log(has_winner);
    let winner=result[3];

    GameOver(has_winner,winner);


}
// document.getElementsByClassName("chess")[0].onclick=luozipk;
// function luozipk(e){
//     if(isOver){
//         addMessage("GameOver Please Restart");
//         return;
//     }
//     if(isBlack){
//         if(chesshistory.length%2===1){
//             ailuozi(e);
//         } else{
//             humanluozi(e);}
//     }else{
//         if(chesshistory.length%2===1){
//             humanluozi(e);
//         } else{
//             ailuozi(e);}
//     }
//
// }
// async function humanluozi(e){
//
//
//
//
//
//
//
//     var x=e.offsetX;
//     var y=e.offsetY;
//
//
//     // console.log(x,y);
//     x=Math.floor((x-start_x/2)/block_chess_width);
//     y=Math.floor((y-start_y/2)/block_chess_height);
//     y=board_height-1-y;
//     console.log("点击坐标为："+x,y);
//
//
//
//     if(x<0||x>board_width||y<0||y>board_height)return;
//     // 已经落子同样g掉
//     if(chessboard[x][y]!==EMPTY){
//         addMessage("不可重复落子");
//         return;
//     }
//     if(chessboard[x][y]===EMPTY){
//         chessboard[x][y]=turn;
//         chesshistory.push({x: x, y: y, color: turn});
//         turn=BLACK+WHITE-turn;
//     }
//     drawChess();
//     addMessage(human_player.name+"落子"+"("+x+","+y+")");
//
//     let result=await human_player.getNextMove(chessboard,x,y);
//
//     // console.log(result);
//     // let nextX=result[0];
//     // let nextY=result[1];
//     // console.log(nextX);
//     // console.log(nextY);
//
//
//     has_winner=result[2];
//     console.log(has_winner);
//     let winner=result[3];
//
//     GameOver(has_winner,winner);
//
// }
// async function ailuozi(e){
//     if(isOver){
//         addMessage("GameOver Please Restart");
//         return;
//     }
//
//
//     addMessage("others is thinking");
//     let x,y;
//     let result=await botPlayer.getNextMove(chessboard,x,y);
//
//     console.log(result);
//     let nextX=result[0];
//     let nextY=result[1];
//     console.log(nextX);
//     console.log(nextY);
//     if(chessboard[nextX][nextY]===EMPTY){
//         chessboard[nextX][nextY]=turn;
//         chesshistory.push({x: nextX, y: nextY, color: turn});
//         turn=BLACK+WHITE-turn;
//     }
//     drawChess();
//
//
//     has_winner=result[2];
//     console.log(has_winner);
//     let winner=result[3];
//     GameOver(has_winner,winner);
//     addMessage("others thinks ok");
//     addMessage(botPlayer.name+"落子"+"("+nextX+","+nextY+")");
//     // document.getElementsByClassName("select")[0].onclick.attr('disabled',true);
// }
//
// //ai落子
// // document.getElementsByClassName("select")[0].onclick= ailuozi(e);
//
//
// //悔棋
//
// // document.getElementsByClassName("backup")[0].onclick=function(){
// //
// //     if(chesshistory.length>0){
// //         // console.log("悔棋前:\n"+chesshistory);
// //         var lastMove = chesshistory.pop(); // 弹出最后一步的落子记录
// //         chessboard[lastMove.x][lastMove.y] = EMPTY; // 从棋盘上移除该棋子
// //         turn = lastMove.color; // 恢复落子颜色
// //         // console.log("悔棋后:\n"+chesshistory);
// //         context.clearRect(0, 0, chess.width, chess.height); // 清空画布
// //         drawChessBoard();
// //         drawChess();
// //         addMessage("悔棋");
// //     }
// //
// // };

var messageBox = document.querySelector('textarea');

// 添加消息到消息框
function addMessage(message) {
    messageBox.value += message + '\n'; // 使用换行符分隔消息
    messageBox.scrollTop = messageBox.scrollHeight; // 滚动到底部
}
function initMessage(message){

     messageBox.value = message + '\n'; // 使用换行符分隔消息
}

