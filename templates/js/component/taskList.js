const ref = Vue.ref;
export default {
  template: `
  <div class="appform">
    <h1>任务列表</h1>
    <el-collapse accordion>

    <el-collapse-item v-for="(t,i) in tableData" :key="i" :title="t.任务名称" :name="i">
    <el-row>
        <h1>{{t.任务名称}}</h1>
    </el-row>
    <el-row>
        <el-col :span="3">任务地点：</el-col>
        <el-col :span="21">{{t.任务地点}}</el-col>
      </el-row>
      <el-row>
        <el-col :span="3">开始日期：</el-col>
        <el-col :span="21">{{t.开始日期}}</el-col>
      </el-row>
      <el-row>
        <el-col :span="3">结束日期：</el-col>
        <el-col :span="21">{{t.结束日期}}</el-col>
      </el-row>
      <el-row>
        <el-col :span="3">任务周期：</el-col>
        <el-col :span="21" v-if="t.周期==0">不重复</el-col>
        <el-col :span="21" v-if="t.周期==1">每天</el-col>
        <el-col :span="21" v-if="t.周期==7">每周</el-col>
        <el-col :span="21" v-if="t.周期==30">每月</el-col>
      </el-row>
      <el-row>
        <el-col :span="3">任务群体：</el-col>
        <el-col :span="21">{{t.任务群体}}</el-col>
      </el-row>
      <el-row>
        <el-col :span="3">任务状态：</el-col>
        <el-col :span="21" v-if="t.当前状态==0">审核不通过</el-col>
        <el-col :span="21" v-if="t.当前状态==1">审核中</el-col>
        <el-col :span="21" v-if="t.当前状态==2">进行中</el-col>
        <el-col :span="21" v-if="t.当前状态==3">已结束</el-col>
      </el-row>
      <el-row>
        <el-col :span="3">任务详情：</el-col>
      </el-row>
      <el-row>
        <el-col :span="24" style="font-size: 10px; padding: 5px;">{{t.任务描述}}</el-col>
      </el-row>
      <el-row style="display: flex; justify-content: center;">
        <el-button type="primary" style="width: 30%;" @click="application(t.任务编号)">
          申请
        </el-button>
      </el-row>
</el-collapse-item>

    </el-collapse>
  </div>
    `,
  props: {
    userdata: Object
  },
  mounted: function () {
    this.getMission()
  },
  data() {
    return {
      page: 0,
      tableData: []
    }
  },
  methods: {
    getMission() {
      axios.post('/back/student/getMission', {
        userdata: this.userdata
      })
        .then((res) => {
          this.tableData = res.data;
        })
    },
    application(missionNo) {
      axios.post('/back/student/postApplication', {
        任务编号: missionNo,
        userdata: this.userdata
      })
        .then((res) => {
          if (res.data) {
            this.$notify.success({
              message: '申请成功',
            })
            this.getMission()
          } else {
            this.$notify.error({
              message: '申请失败',
            })
          }
        })
    }
  }
}

