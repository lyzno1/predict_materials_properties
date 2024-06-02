import elcelUp from "../elcelUp/elcelUp.js"

export default {
  template: `
  <el-row style="width: 100%;">

        <el-col :span="12">
            <el-button type="primary" @click="addOne = true" text bg>
                <el-icon>
                    <Plus />
                </el-icon>
                手动导入
            </el-button>
        </el-col>

        <el-col :span="12">
          <el-row>
            <elcel-up @get-data-list="get"></elcel-up>
            <a href="/data/student.xlsx" download>模板下载</a>
          </el-row>
        </el-col>

  </el-row>

  <el-dialog v-model="addOne" title="手动导入">
        <el-form :model="temp">
            <el-form-item label="类别">
                <el-input v-model="temp.类别" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="学院">
                <el-input v-model="temp.学院" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="校区">
                <el-input v-model="temp.校区" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="姓名">
                <el-input v-model="temp.姓名" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="学号">
                <el-input v-model="temp.学号" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="班级">
                <el-input v-model="temp.班级" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="手机号">
                <el-input v-model="temp.手机号" autocomplete="off"></el-input>
            </el-form-item>
        </el-form>
        <template #footer>
            <span class="dialog-footer">
                <el-button @click="addOne = false">取消</el-button>
                <el-button type="primary" @click="addOneStudent()">
                    确认
                </el-button>
            </span>
        </template>
    </el-dialog>
    
  <el-table height="calc(100vh - 132px)" stripe border :data="tableData" style="width: 100%"
        :cell-style="{ textAlign: 'center' }" :header-cell-style="{ 'text-align': 'center' }">
        <el-table-column prop="类别" label="类别"></el-table-column>
        <el-table-column prop="学院" label="学院"></el-table-column>
        <el-table-column prop="校区" label="校区"></el-table-column>
        <el-table-column prop="姓名" label="姓名"></el-table-column>
        <el-table-column prop="学号" label="学号"></el-table-column>
        <el-table-column prop="班级" label="班级"></el-table-column>
        <el-table-column prop="手机号" label="手机号"></el-table-column>
        <el-table-column label="操作">
            <template v-slot="scope" style="display: flex; flex-direction: column;">
                <el-button type="primary" plain @click="openChange(scope.row.学号)">修改</el-button>
                <el-popconfirm confirm-button-text="是" cancel-button-text="否" title="是否删除本行数据?"
                    @confirm="del(scope.row.学号)">
                    <template #reference>
                        <el-button type="danger" plain>删除</el-button>
                    </template>
                </el-popconfirm>
                <el-popconfirm confirm-button-text="是" cancel-button-text="否" title="是否重置密码?"
                    @confirm="reSet(scope.row.学号)">
                    <template #reference>
                        <el-button type="danger" plain>重置密码</el-button>
                    </template>
                </el-popconfirm>
            </template>
        </el-table-column>
    </el-table>

    <el-dialog v-model="updataOne" title="修改">
        <el-form :model="temp">
            <el-form-item label="类别">
                <el-input v-model="temp.类别" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="学院">
                <el-input v-model="temp.学院" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="校区">
                <el-input v-model="temp.校区" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="姓名">
                <el-input v-model="temp.姓名" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="学号">
                <el-input v-model="temp.学号" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="班级">
                <el-input v-model="temp.班级" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="手机号">
                <el-input v-model="temp.手机号" autocomplete="off"></el-input>
            </el-form-item>
        </el-form>
        <template #footer>
            <span class="dialog-footer">
                <el-button @click="updataOne = false">取消</el-button>
                <el-button type="primary" @click="change()">
                    确认
                </el-button>
            </span>
        </template>
    </el-dialog>
  `,
  components: {
    elcelUp,
  },
  data() {
    return {
      temp: {
        类别: '',
        学院: '',
        校区: '',
        姓名: '',
        学号: '',
        班级: '',
        手机号: ''
      },
      rules: {
        required: true,
        message: '请输入',
        trigger: 'blur',
      },
      tempUploadData: [],
      tableData: [],
      addOne: false,
      updataOne: false,
      changeId: null,

      back: '/back/admin/student/'
    }
  },
  mounted: function () {
    this.getData();
  },
  methods: {
    get(data) {
      this.tempUploadData = data
      this.tableData = []
      this.uploadData()
    },
    openChange(id) {
      console.log(id);
      this.changeId = this.tableData.indexOf(this.tableData.find(item => item.学号 == id))
      this.temp = this.tableData[this.changeId]
      console.log(this.temp);
      this.updataOne = true
    },
    change() {
      axios.post(this.back + 'studentChange', { studentData: this.temp })
        .then((res) => {
          if (res.data) {
            alert('修改成功')
          } else {
            alert("修改失败");
          }
          this.getData()
        })
        .catch(error => {
          alert("修改失败");
          console.error(error);
        });
      this.changeId = null
      this.temp = {
        类别: '',
        学院: '',
        校区: '',
        姓名: '',
        学号: '',
        班级: '',
        手机号: ''
      }
      this.updataOne = false

    },
    del(id) {
      axios.post(this.back + 'studentDel', { id: id })
        .then((res) => {
          if (res.data) {
            alert('删除成功')
          } else {
            alert("删除失败");
          }
          this.getData()
        })
        .catch(error => {
          alert("删除失败");
          console.error(error);
        });
    },
    getData() {
      axios.post(this.back + 'checkstudent')
        .then((res) => {
          this.tableData = res.data
        })
    },
    uploadData() {
      axios.post(this.back + 'studentReg', {
        studentData: this.tempUploadData,
      })
        .then((res) => {
          this.tempUploadData = [];
          this.getData()
        });
    },
    addOneStudent() {
      this.tempUploadData = this.temp
      this.uploadData()
      this.temp = {
        类别: '',
        学院: '',
        校区: '',
        姓名: '',
        学号: '',
        班级: '',
        手机号: ''
      }
      this.addOne = false
    },
    reSet(id) {
      axios.post(this.back + 'studentReset', { id: id })
        .then((res) => {
          if (res.data) {
            alert('重置成功')
          } else {
            alert("重置失败");
          }
          this.getData()
        })
        .catch(error => {
          alert("重置失败");
          console.error(error);
        });
    }
  }
}