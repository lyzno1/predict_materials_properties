const { createApp } = Vue
const ref = Vue.ref;
const toRefs = Vue.toRefs;

import teacherShow from './component/teacherShow/teacherShow.js'
import auditTemplates from './component/auditTemplates/auditTemplates.js'
import childrenShow from './component/childreShow/childreShow.js'
const app = createApp({

    components: {
        childrenShow,
        teacherShow,
        auditTemplates
    },
    data() {
        return {
            page: 3,
        }
    },
    methods: {
        pageChange(id) {
            this.page = id
        }
    }
})

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}
app.use(ElementPlus)
app.mount('#app')

