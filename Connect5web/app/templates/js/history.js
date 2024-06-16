Vue.config.productionTip =false;
new Vue({
    el: '#app',
    data: {
        address:"http://127.0.0.1:5500/",
        userInfo: {
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

        historyshow: true,
         historyData: [{
              id:-1,
              player1_name: '',
              player2_name: '',
              winner_name: '',
              created_at: '',
              ended_at: ''
            }]

    },
    mounted() {
        this.userInfo.name=this.getCookie('username');
        this.load();
    },
    methods: {
        load(){
            // alert(document.cookie);


            axios.post(this.address+"history/getHistory",this.userInfo)
                .then(res => {

                    if(res.data.message==='Entry not found'){
                        this.historyData=null;
                    }else{
                        console.log(res.data);
                        this.historyData=res.data;
                    }


                })
                .catch(error => {
                    console.error("Error: ", error);
                });

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
        userinfo(){

            window.location.href = this.address+"user.html";
        },
        historyinfo(){
             window.location.href = this.address+"history.html";
        },
        PKWithRenju(){

             window.location.href = this.address+"renju.html";
        },
        PKWithOthers(){
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
        handleShow(index, row) {
            let id= row.id;
            this.setCookie('memory_id',id);
            window.location.href = this.address+"memory.html";
            // axios.post(this.address+"history/getMemory", {id:id})
            //     .then(res => {
            //         console.log(res.data);
            //
            //
            //     })
            //     .catch(error => {
            //         console.error("Error: ", error);
            //     });

             // this.load();
        // console.log(index, row);
      },
        handleDelete(index, row) {
            let id= row.id;
            axios.post(this.address+"history/deleteHistory", {id:id})
                .then(res => {
                    console.log(res.data);

                })
                .catch(error => {
                    console.error("Error: ", error);
                });

             this.load();
        console.log(index, row);
      }


    }
});