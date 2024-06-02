import elcelUp from "../elcelUp/elcelUp.js"

export default {
  template:`
  <el-row style="width: 100%;">

        <el-col :span="8">
            <el-button type="primary" @click="addOne = true" text bg>
                <el-icon>
                    <Plus />
                </el-icon>
                手动导入
            </el-button>
        </el-col>

        <el-col :span="8"><elcel-up @get-data-list="get"></elcel-up></el-col>

        <el-col :span="8">
            <el-button @click="uploadData" :disabled="upload" type="primary" text bg>
                <el-icon>
                    <Upload />
                </el-icon>
                上传
            </el-button>
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
                <el-button type="primary" @click="addOneTeacher()">
                    确认
                </el-button>
            </span>
        </template>
    </el-dialog>

    <el-table stripe height="calc(100vh - 122px)" border :data="tableData" style="width: 100%"
        :cell-style="{ textAlign: 'center' }" :header-cell-style="{ 'text-align': 'center' }">
        <el-table-column prop="类别" label="类别"></el-table-column>
        <el-table-column prop="学院" label="学院"></el-table-column>
        <el-table-column prop="校区" label="校区"></el-table-column>
        <el-table-column prop="姓名" label="姓名"></el-table-column>
        <el-table-column prop="学号" label="学号"></el-table-column>
        <el-table-column prop="班级" label="班级"></el-table-column>
        <el-table-column prop="手机号" label="手机号"></el-table-column>
        <el-table-column label="操作">
            <template v-slot="scope">
                <el-button type="primary" plain @click="openChange(scope.row.学号)">修改</el-button>
                <el-popconfirm confirm-button-text="是" cancel-button-text="否" icon-color="#626AEF" title="是否删除本行数据?"
                    @confirm="del(scope.row.学号)">
                    <template #reference>
                        <el-button type="danger" plain>删除</el-button>
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
      tableData: [],
      upload: true,
      addOne: false,
      updataOne: false,
      changeId: null,
    }
  },
  watch: {
    tableData(newData) {
      if (newData) {
        this.upload = false
      } else {
        this.upload = true
      }
    }
  },
  methods: {
    get(data) {
      this.tableData = data
      console.log(this.tableData);
    },
    openChange(id) {
      this.changeId = this.tableData.indexOf(this.tableData.find(item => item.学号 == id))
      this.temp = this.tableData[this.changeId]
      this.updataOne = true
    },
    change() {
      this.tableData[this.changeId] = this.temp
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
      this.tableData.splice(this.tableData.indexOf(this.tableData.find(item => item.学号 == id)), 1);
    },
    addOneTeacher() {
      this.tableData.push(this.temp)
      this.addOne = false
      this.temp = {
        类别: '',
        学院: '',
        校区: '',
        姓名: '',
        学号: '',
        班级: '',
        手机号: ''
      }
    },
    uploadData() {
      axios.post('/page/php/mysqlTool.php', {
        data1: this.tableData,
      });
    }
  },

}