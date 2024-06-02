export default {
  template: `
    <el-container id="main-container">
    <el-aside id="main-container-aside">
      <div v-for="(p,id) in projects" :key='id+1'>
        <el-button @click="projectId=id+1;project=p;getApplication(p.任务编号)" :class="{projectId: id==projectId}" text>
          {{p.任务名称}}
        </el-button>
      </div>
    </el-aside>

    <el-main id="main-container-main">
      <div v-if="projectId!==0">
        <div id="main-container-main-project">
          <el-row style="margin-bottom: 5px;">
            <span style="font-size: 40px; font-weight: bold; color: black;">任务详情</span>
          </el-row>
          <el-row>
            <el-col :span="3">任务名称：</el-col>
            <el-col :span="21">{{project.任务名称}}</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">任务地点：</el-col>
            <el-col :span="21">{{project.任务地点}}</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">开始日期：</el-col>
            <el-col :span="21">{{project.开始日期}}</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">结束日期：</el-col>
            <el-col :span="21">{{project.结束日期}}</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">任务周期：</el-col>
            <el-col :span="21" v-if="project.周期==0">不重复</el-col>
            <el-col :span="21" v-if="project.周期==1">每天</el-col>
            <el-col :span="21" v-if="project.周期==7">每周</el-col>
            <el-col :span="21" v-if="project.周期==30">每月</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">任务群体：</el-col>
            <el-col :span="21">{{project.任务群体}}</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">任务状态：</el-col>
            <el-col :span="21" v-if="project.当前状态==0">审核不通过</el-col>
            <el-col :span="21" v-if="project.当前状态==1">审核中</el-col>
            <el-col :span="21" v-if="project.当前状态==2">进行中</el-col>
            <el-col :span="21" v-if="project.当前状态==3">已结束</el-col>
          </el-row>
          <el-row>
            <el-col :span="3">任务详情：</el-col>
          </el-row>
          <el-row>
            <el-col :span="24" style="font-size: 10px; padding: 5px;">{{project.任务描述}}</el-col>
          </el-row>

        </div>

        <el-table v-if='projectId!==0 && applyForList.length>0' :data="applyForList" border stripe
          style="width: 100%;margin-top: 20px;">
          <el-table-column prop="姓名" label="申请人" width="100"></el-table-column>
          <el-table-column prop="学号" label="学号" width="110"></el-table-column>
          <el-table-column prop="学院" label="学院" width="120"></el-table-column>
          <el-table-column prop="班级" label="班级"></el-table-column>
          <el-table-column label="操作" width="160">
            <template v-slot="scope">
              <el-button type="primary" @click="agreeApp(scope.row.任务编号,true)">同意</el-button>
              <el-button type="info" @click="agreeApp(scope.row.任务编号,false)">拒绝</el-button>
            </template>
          </el-table-column>

        </el-table>
      </div>
    </el-main>
  </el-container>
    `,
  props: {
    username: String
  },
  data() {
    return {
      project: null,
      projectId: 0,
      projects: null,
      applyForList: []
    }
  },
  mounted: function () {
    this.getProjectList();
  },
  methods: {
    getProjectList() {
      axios.post('/back/teacher/getProject', {
        发布人: this.username
      })
        .then((res) => {
          this.projects = res.data
        })
    },
    getApplication(missionNum){
      axios.post('/back/teacher/getApplication',{
        任务编号:missionNum
      })
      .then((res)=>{
        this.applyForList=res.data
      })
    },
    agreeApp(missionNum,ans) {
      axios.post('/back/teacher/agreeApp',{
        任务编号:missionNum,
        ans:ans
      })
      .then((res)=>{
        if (res.data) {
          this.$notify.success({
            message: '审核成功',
          })
          this.getApplication(missionNum)
        } else {
          this.$notify.error({
            message: '审核失败',
          })
        }
      })
    }
  }
}