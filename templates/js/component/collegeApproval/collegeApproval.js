export default {
  template: `
  <el-container id="main-container">
            <el-aside id="main-container-aside">
              <div v-for="(project,id) in projects" :key='id+1'>
                <el-button @click="projectId=id+1;getProject(id)" :class="{projectId:id+1==projectId}" text>
                  {{project.任务名称}}
                </el-button>
              </div>
            </el-aside>

            <el-main id="main-container-main">
              <div v-if="projectId!==0 ">

              <el-row style="margin-bottom: 5px;">
              <span style="font-size: 40px; font-weight: bold; color: black;">任务详情</span>
            </el-row>
            <el-row>
              <el-col :span="3">任务名称：</el-col>
              <el-col :span="21">{{approval.任务名称}}</el-col>
            </el-row>
            <el-row>
              <el-col :span="3">任务地点：</el-col>
              <el-col :span="21">{{approval.任务地点}}</el-col>
            </el-row>
            <el-row>
              <el-col :span="3">开始日期：</el-col>
              <el-col :span="21">{{approval.开始日期}}</el-col>
            </el-row>
            <el-row>
              <el-col :span="3">结束日期：</el-col>
              <el-col :span="21">{{approval.结束日期}}</el-col>
            </el-row>
            <el-row>
              <el-col :span="3">任务周期：</el-col>
              <el-col :span="21" v-if="approval.周期==0">不重复</el-col>
              <el-col :span="21" v-if="approval.周期==1">每天</el-col>
              <el-col :span="21" v-if="approval.周期==7">每周</el-col>
              <el-col :span="21" v-if="approval.周期==30">每月</el-col>
            </el-row>
            <el-row>
              <el-col :span="3">任务群体：</el-col>
              <el-col :span="21">{{approval.任务群体}}</el-col>
            </el-row>
            <el-row>
              <el-col :span="3">任务详情：</el-col>
            </el-row>
            <el-row>
              <el-col :span="24" style="font-size: 10px; padding: 5px;">{{approval.任务描述}}</el-col>
            </el-row>

                <div style="position: relative; height: 100vh;">
                  <el-button type="primary" size="large"
                    style="position: absolute; top: 20%; left: 40%; transform: translateX(-50%);"
                    @click="agree">通过</el-button>
                  <el-button type="danger" size="large"
                    style="position: absolute; top: 20%; left: 60%; transform: translateX(-50%);"
                    @click="refuse">打回</el-button>
                </div>

              </div>
            </el-main>
          </el-container>
  `,
  props: {
    username: String
  }
  ,
  data() {
    return {
      projects: null,
      projectId: 0,
      approval: []
    }
  },
  mounted: function () {
    this.getProjectList();
  },
  methods: {
    getProjectList() {
      axios.post('/back/teacher/getApproval', {
        工号: this.username
      })
        .then((res) => {

          this.projects = res.data
        })
    },
    /**
     * @description: 获取任务详情
     * @param {*} id 任务编号-主键
     * @return {*}
     */
    getProject(id) {
      this.approval = this.projects[id]
    },
    agree() {
      // 在这里编写同意事件的处理代码
      axios.post('/back/teacher/approval', {
        approval: this.approval,
        username: this.username,
        ans: true
      })
        .then((res) => {
          if (res.data) {
            this.$notify.success({
              message: '审核成功',
            })
            this.getProjectList()
          } else {
            this.$notify.error({
              message: '审核失败',
            })
          }
        })
    },
    refuse() {
      // 在这里编写拒绝事件的处理代码
      axios.post('/back/teacher/approval', {
        approval: this.approval,
        username: this.username,
        ans: false
      })
        .then((res) => {
          if (res.data) {
            this.$notify.success({
              message: '打回成功',
            })
            this.getProjectList()
          } else {
            this.$notify.error({
              message: '打回失败',
            })
          }
        })
    },
  }
}
