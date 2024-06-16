Vue.config.productionTip =false;
new Vue({
    el: '#app',
    data: {
        address:"http://127.0.0.1:5500/",
        imageUrl:'',
        userInfo: {
            id:-1,
            name: '',
            email: 'john@example.com',
            avatar : 'image/login.jpg',
            registration_time:'2024.3.6',
            state: 'online'
        },
         centerDialogVisible: false,
         tableData: [{
              player1: '',
              player2: '',
              game_id:' ',
             name:''
            }],
        loading: true,

        historyshow: false,
         historyData: [{
              name: '2016-05-02',
              winner_name: '王小虎',
              created_at: '上海市普陀区金沙江路 1518 弄',
              ended_at: '上海市普陀区金沙江路 1518 弄'
            }, {
              name: '2016-05-02',
              winner_name: '王小虎',
              created_at: '上海市普陀区金沙江路 1518 弄',
              ended_at: '上海市普陀区金沙江路 1518 弄'
            }, {
              name: '2016-05-02',
              winner_name: '王小虎',
              created_at: '上海市普陀区金沙江路 1518 弄',
              ended_at: '上海市普陀区金沙江路 1518 弄'
        }],
        updateUserShow:false



    },
    mounted() {
        this.userInfo.name=this.getCookie('username');
        this.load();
    },
    methods: {

        load(){
            // alert(document.cookie);
            historyshow=false;
            // this.clearCookies();
            axios.post(this.address+"users/getByName",this.userInfo)
                .then(res => {
                    console.log(res.data);
                     this.userInfo.avatar = res.data.avatar;
                     if( this.userInfo.avatar==null){
                         this.userInfo.avatar="image/login.jpg";
                     }
                      this.userInfo.id=res.data.id;
                     // alert(res.data.id);
                      this.userInfo.name = res.data.name;
                      this.userInfo.email = res.data.email;
                      this.userInfo.registration_time = res.data.registration_time;
                      this.userInfo.state = res.data.state;
                      this.userInfo.avatar = res.data.avatar;
                })
                .catch(error => {
                    console.error("Error: ", error);
                });

        },
        updateUserInfo(){
            this.updateUserShow=true;

        },
        sendUserInfo(){
             axios.post(this.address+"users/updateUserById",this.userInfo)
                .then(res => {
                    console.log(res.data);
                })
                .catch(error => {
                    console.error("Error: ", error);
                });
             this.updateUserShow=false;
             this.load();
        },
        setCookie(name, value, days) {
            let expires = "";
            if (days) {
                let date = new Date();
                date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
                expires = "; expires=" + date.toUTCString();
            }
            document.cookie = name + "=" + (value || "") + expires + "; path=/";
        },

        getCookie(name) {
            let nameEQ = name + "=";
            let ca = document.cookie.split(';');
            for (let i = 0; i < ca.length; i++) {
                let c = ca[i];
                while (c.charAt(0) === ' ') c = c.substring(1, c.length);
                if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
            }
            return null;
        },

        parseCookies() {
            let cookies = {};
            document.cookie.split(';').forEach(cookie => {
                let [name, value] = cookie.split('=').map(c => c.trim());
                cookies[name] = value;
            });
            return cookies;
        },

        clearCookies() {
            let cookies = document.cookie.split(";");
            for (let i = 0; i < cookies.length; i++) {
                let cookie = cookies[i];
                let eqPos = cookie.indexOf("=");
                let name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
                document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
            }
        },
        userinfor(){
             window.location.href = this.address+"user.html";
        },
        historyinfo(){
             window.location.href = this.address+"history.html";
        },
        PKWithRenju(){
             this.historyshow=false;
             window.location.href = this.address+"renju.html";
        },
        PKWithOthers(){
            this.historyshow=false;
            this.centerDialogVisible = true;
            axios.post(this.address+"pk/matchPlayers",this.userInfo)
                .then(res => {
                    console.log(res.data);



                    this.tableData[0].player1 = res.data.player1;
                    this.tableData[0].player2 = res.data.player2;
                    this.tableData[0].game_id = res.data.game_id;
                    this.tableData[0].name = res.data.name;
                    this.loading = false;

                    this.setCookie('player1',res.data.player1);
                    this.setCookie('player2',res.data.player2);
                    this.setCookie('game_id',res.data.game_id);
                    this.setCookie('name',res.data.name);
                    window.location.href = this.address+'PK.html';
                })
                .catch(error => {
                    console.error("Error: ", error);
                });

        },
        handleEdit(index, row) {
        console.log(index, row);
      },
        handleDelete(index, row) {
        console.log(index, row);
      },//上传头像
      handleAvatarSuccess(res, file) {


        this.imageUrl = URL.createObjectURL(file.raw);
        this.userInfo.avatar = res;
        // console.log(res);

      },
      beforeAvatarUpload(file) {
        const isJPG = file.type === 'image/jpeg';
        const isLt2M = file.size / 1024 / 1024 < 2;

        if (!isJPG) {
          this.$message.error('上传头像图片只能是 JPG 格式!');
        }
        if (!isLt2M) {
          this.$message.error('上传头像图片大小不能超过 2MB!');
        }
        return isJPG && isLt2M;
      }


    }
});