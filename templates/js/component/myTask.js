
export default {
  template: `
  <div style="margin-bottom: 20px;">
        <span style="font-size: 30px;">我的任务</span>
    </div>
    <el-collapse accordion>

        <el-collapse-item v-for="(t,i) in tableData" :key="i" :name="i" @click="missionId=t.任务编号,getFiles()">
            <template #title>
                <el-row>
                    <el-col :span="20">
                        <h1>{{t.任务名称}}</h1>
                    </el-col>
                    <el-col :span="4">
                        <el-tag v-if="t.状态==1 && t.当前状态==2">进行中</el-tag>
                        <el-tag type="danger" v-if="t.状态==1 && t.当前状态==3">已结束</el-tag>
                        <el-tag type="info" v-if="t.状态==0">审核中</el-tag>
                        <el-tag type="danger" v-if="t.状态==-1">未通过</el-tag>
                    </el-col>
                </el-row>
            </template>
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
                <el-col :span="3">任务详情：</el-col>
            </el-row>
            <el-row>
                <el-col :span="24" style="font-size: 10px; padding: 5px;">{{t.任务描述}}</el-col>
            </el-row>
            <el-row style="display: flex; justify-content: center;">
                <el-button type="primary" style="width: 30%;" @click="centerDialogVisible=true">
                    提交附件
                </el-button>
            </el-row>

            <el-table v-if="oldFiles!=[]" height="calc(200px)" stripe border :data="oldFiles" style=" margin-top:20px;width: 100%"
                :cell-style="{ textAlign: 'center' }" :header-cell-style="{ 'text-align': 'center' }">
                <el-table-column prop="文件名" label="文件名"></el-table-column>
                <el-table-column prop="时间" label="提交时间"></el-table-column>
                <el-table-column label="操作">
                    <template v-slot="scope" style="display: flex; flex-dicrection: column;">
                        <el-button type="primary" plain @click=" downloadFile(scope.row.地址)">下载</el-button>

                        <el-button type="danger" plain
                          @click="del(scope.row.任务编号,scope.row.学号,scope.row.时间,scope.row.文件名)">
                            删除
                          </el-button>

                    </template>
                </el-table-column>
            </el-table>
        </el-collapse-item>

    </el-collapse>

    <el-dialog v-model="centerDialogVisible" title="提交附件" center>

        <el-upload drag multiple ref="upload" action="#" :on-change="handleChange" :auto-upload="false">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
                拖动文件至此或者 <em>点击上传文件</em>
            </div>
            <template #tip>
                <div class="el-upload__tip">
                    files with a size less than 20mb
                </div>
            </template>
        </el-upload>>

        <el-row>
            <el-button @click="submit" type="primary"
                style="border: 1px solid #409EFF; border-radius: 4px; margin-top: 50px; margin-left: 48%; size:large">
                提交
            </el-button>
        </el-row>

    </el-dialog>
    
    `,
  props: {
    userdata: Object
  },
  data() {
    return {
      page: 0,
      tableData: [],
      missionId: null,

      oldFiles: [],

      fileList: [],

      tableData1: [
      ],
      activeName: 'first',
      centerDialogVisible: false,
    }
  },
  mounted: function () {
    this.getApplication()
  },
  methods: {
    getApplication() {
      axios.post('/back/student/getApplication', {
        userdata: this.userdata
      })
        .then((res) => {
          this.tableData = res.data
        })
    },
    handleChange(file, fileList) {
      this.fileList = fileList
    },
    getFiles() {
      this.oldFiles = []
      axios.post('/back/student/getFiles', {
        学号: this.userdata.No,
        任务号: this.missionId
      })
        .then((res) => {
          this.oldFiles = res.data;
        })
    },
    submit() {
      /* this.$refs.upload.submit() */
      var f = new FormData()
      f.append('任务号', this.missionId)
      f.append('学号', this.userdata.No)
      for (var i in this.fileList) {
        f.append('file[]', this.fileList[i].raw)
      }

      axios.post('/back/student/postFile', f)
        .then((res) => {
          if (res.data) {
            this.$notify.success({
              message: '提交成功',
            })
            this.$ref.upload.clearFiles();
            this.centerDialogVisible = false;
          } else {
            this.$notify.error({
              message: '提交失败',
            })
          }
        })

    },
    del(ID, No, date, file) {
      axios.post("/back/student/delFile",{
        任务号: ID,
        学号: No,
        时间: date,
        文件名: file
      })
        .then((res) => {
          if (res.data) {
            this.$notify.success({
              message: '删除成功',
            })
          } else {
            this.$notify.error({
              message: '删除失败',
            })
          }
          this.getFiles()
        })
    },
    downloadFile(url) {
      window.open(url, '_self')
    }
  }
}

