const { createApp } = Vue
const ref = Vue.ref;
const toRefs = Vue.toRefs;

//这里添加组件
//import ....
import checkApplyFor from "./component/checkApplyFor/checkApplyFor.js";
import evaluate from "./component/evaluate/evaluate.js"
import newProject from './component/newProject/newProject.js'
import collegeApproval from './component/collegeApproval/collegeApproval.js';
const app = createApp({
    components: {
        checkApplyFor,
        evaluate,
        newProject,
        collegeApproval,
    },
    mounted: function () {
        this.username = window.sessionStorage.getItem('username');
        this.password = window.sessionStorage.getItem('password');
        this.getData();


    },
    data() {
        return {
            page: 0,
            username: null,
            password: null,
            userdata: [],
            user: false,
            newPassword: false,
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
            }


        }
    },
    methods: {





        pageChange(id) {
            this.page = id
        },
        getData() {
            axios.post('/back/teacher/getData', {
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
            window.open('./tlogin.html', '_self');
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
        }
    }
})

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}
app.use(ElementPlus)
app.mount('#app')