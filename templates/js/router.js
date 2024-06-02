const router=createRouter({
    history:createWebHistory(),
    routes:[
        {
            component:()=>import('../pages/person.vue'),
            name:'person',
            path:'/person'

        },

        {
            component:()=>import('../pages/student.vue'),
            name:'student',
            path:'/student'

        }

    ]

})
export default router