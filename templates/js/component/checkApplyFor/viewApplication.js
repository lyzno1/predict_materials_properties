const ref = Vue.ref;

export default {
    template: `
    <div class="appform">
    <el-text class="lookat">申请内容</el-text>
    <el-table :data="tableData" border stripe style="width: 70%">
    <el-table-column prop="applicant" label="申请人" width="180" />
    <el-table-column prop="academy" label="学院" width="180" />
    <el-table-column prop="content" label="申请内容" />
    <el-table-column prop="operation" label="操作">
    <span slot-scope="scope">
      <el-button type="primary" size="medium" @click="agreeApp">同意</el-button>
      <el-button type="info" size="medium" @click="refuseApp">拒绝</el-button>
    </span>
  </el-table-column>
  
  </el-table>
  </div>

    `,
    data() {
        return {
            page: 0
            ,tableData: [
                {
                  applicant: '王欣浩',
                  academy: '计算机学院',
                  content: '机房卫生打扫',
                },
                {
                    applicant: '刘震宇',
                    academy: '计算机学院',
                    content: '饭堂卫生执勤',
                  }, {
                    applicant: '张三',
                    academy: '其他学院',
                    content: '搬书',
                  },
              ]
        }
    },
    methods: {
        agreeApp() {
        //处理同意申请之后的事件  
        }
        
        ,refuseApp() {
        //处理拒绝申请之后的事件   
        }

        ,pageChange(id) {
            this.page = id
        }
    }
}

