
new Vue({
    el: '#app',
    data() {
        return {
            address:'http://127.0.0.1:5500/',
            loginForm: {
                username: 'admin',
                password: 'admin'
            },
            registerForm: {
                username: '',
                password: '',
                confirmPassword: '',
                email:''
            },
            rules: {
                username: [{ required: true, message: '请输入账号', trigger: 'blur' }],
                password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
            },
            registerRules: {
                username: [{ required: true, message: '请输入账号', trigger: 'blur' }],
                password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
                confirmPassword: [
                    { required: true, message: '请确认密码', trigger: 'blur' },
                    {
                        validator: (rule, value, callback) => {
                            if (value !== this.registerForm.password) {
                                callback(new Error('两次输入密码不一致'));
                            } else {
                                callback();
                            }
                        },
                        trigger: 'blur'
                    }
                ]
            },
            showRegisterForm: false,
            showLoginForm: true
        };
    },
    methods: {
        submitForm(formName) {
            if (formName === 'loginForm') {
                this.$refs.loginForm.validate((valid) => {
                    if (valid) {
                        this.login();
                    }
                });
            } else if (formName === 'registerForm') {
                this.$refs.registerForm.validate((valid) => {
                    if (valid) {
                        this.register();
                    }
                });
            }
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
        login() {
            axios.post(this.address+"auths/login", this.loginForm)
                .then(response => {
                    console.log(response.data);
                    if (response.data.message === 'login ok') {
                        // this.clearCookies();
                        this.setCookie('username',this.loginForm.username);
                        // alert(document.cookie);
                        window.location.href = this.address+"user.html";
                    } else {
                        this.$message.error(response.data.message);
                    }
                })
                .catch(error => {
                    console.error("Error: ", error);
                });
        },
        register() {
            axios.post(this.address+"auths/register", this.registerForm)
                .then(response => {
                    console.log(response.data);
                    if (response.data.message === "register ok") {
                        this.$message.success("注册成功，请登录。");
                    } else {
                        this.$message.error(response.data.message);
                    }
                })
                .catch(error => {
                    console.error("Error: ", error);
                });
        }
    }
});