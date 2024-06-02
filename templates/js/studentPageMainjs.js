const { createApp } = Vue
const ref = Vue.ref;
const toRefs = Vue.toRefs;

//这里添加组件
//import ....
import myTask from './component/myTask.js'
import taskList from './component/taskList.js';
import attachments from './component/attachments.js';
const app = createApp({
    components: {
        taskList
        , myTask
        , attachments
    },
    mounted: function () {
        this.username = window.sessionStorage.getItem('username');
        this.password = window.sessionStorage.getItem('password');
        this.getData();
    },
    data() {
        return {
            page: 2,
            username: null,
            password: null,
            userdata: null,
            user: false,
            newPassword: false,
            tempPhone: null,
            formData: {
                password: '',
                confirmPassword: ''
            },

            formRules: {
                password: [
                    { required: true, message: '请输入新密码', trigger: 'blur' }
                ],
                confirmPassword: [
                    { required: true, message: '请确认新密码', trigger: 'blur' },
                    {
                        validator: (rule, value, callback) => {
                            if (value !== this.formData.password) {
                                callback(new Error('两次输入的密码不一致'))
                            } else {
                                callback()
                            }
                        },
                        trigger: 'blur'
                    }
                ]
            },

        }
    },
    methods: {
        pageChange(id) {
            this.page = id
        },
        getData() {
            axios.post('/back/student/getData', {
                username: this.username,
                password: this.password
            })
                .then((res) => {
                    this.userdata = res.data;
                })
        },
        handleClick() {
            this.userdata = null;
            this.username = null;
            window.sessionStorage.clear()
            window.open('./stlogin.html', '_self');
        },
        upNewPassword() {
            axios.post('/back/teacher/newPassword', {
                username: this.username,
                userpassword: this.formData.password
            })
                .then((res) => {
                    if (res.data) {
                        alert('修改成功')
                        this.formData.password = ''
                        this.formData.confirmPassword = ''
                        this.handleClick()
                    }
                    else {
                        alert('修改失败')
                    }

                })
        },
        upPhone() {
            axios.post('/back/student/upPhone', {
                学号: this.username,
                手机号: this.tempPhone
            })
                .then((res) => {
                    if (res.data) {
                        alert('修改成功')
                        this.getData()
                        this.user = false
                    }
                    else {
                        alert('修改失败')
                    }
                })
        }
    }
})

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}
app.use(ElementPlus)
app.mount('#app')