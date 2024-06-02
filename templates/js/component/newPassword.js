
export default {
  template: `
    <div>
    <   "dialogVisible" v-model="showDialog" @close="handleClose">
      <el-form ref="form" :model="formData" :rules="formRules">
        <el-form-item label="新密码" prop="password">
          <el-input type="password" v-model="formData.password" show-password></el-input>
        </el-form-item>
        <el-form-item label="确认密码" prop="confirmPassword">
          <el-input type="password" v-model="formData.confirmPassword" show-password></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
      <span class="dialog-footer">
        <el-button @click="showDialog = false">
          取消
        </el-button>
        <el-button type="primary" @click="showDialog = false">
          确认
        </el-button>
      </span>
    </template>
    </el-dialog>
  </div>
      `,
  data() {
    return {
      dialogVisible: false,
      showDialog: true,
      formData: {
        password: '',
        confirmPassword: ''
      },
      formRules: {
        password: [
          { required: true, message: '请输入新密码', trigger: 'blur' }
        ],
        confirmPassword: [
          { required: true, message: '请确认新密码', trigger: 'blur' },
          {
            validator: (rule, value, callback) => {
              if (value !== this.formData.password) {
                callback(new Error('两次输入的密码不一致'))
              } else {
                callback()
              }
            },
            trigger: 'blur'
          }
        ]
      }
    }
  },

  methods: {
    pageChange(id) {
      this.page = id
    }
    ,
    // showDialog() {
    //     this.dialogVisible = true
    //   },
    //   submitForm() {
    //     this.$refs.form.validate(valid => {
    //       if (valid) {
    //         // 验证通过，可以提交表单
    //         console.log(this.formData)
    //         this.dialogVisible = false
    //       }
    //     })
    //   },
    handleClose() {
      //清空表单数据
      this.formData.password = ''
      this.formData.confirmPassword = ''
    }
  }
}

