const { createApp, ref } = Vue;

// 创建 Vue 应用
const app = createApp({
    data() {
        return {
            form: {
                atom1: '',
                atom2: ''
            },
            dialogVisible: false,
            errorMessage: ''
        };
    },
    methods: {
        predictDistance() {
            const { atom1, atom2 } = this.form;
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ atom1, atom2 }),
            })
            .then(response => response.json())
            .then(data => {
                alert(`预测的原子间距离为: ${data.distance} Å`);
            })
            .catch((error) => {
                console.error('Error:', error);
                this.errorMessage = '发生错误，请稍后再试';
                this.dialogVisible = true;
            });
        }
    },
    setup() {
        const message = ref('Hello vue!')
        return {
            message
        }
    }
});

// 在 Vue 实例创建之前注册 ElementPlus
app.use(ElementPlus);

// 挂载 Vue 应用
app.mount('#app');
