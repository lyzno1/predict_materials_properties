
export default {
    created() {
      const taskId = this.$route.params.taskId
      // 根据任务ID获取任务详情数据
      // ...
    }
    ,template: `
  <el-text>提交任务附件</el-text>
  <div>
  <div class="form-row">
  <el-text class="mx-2">任务选择</el-text>
  <el-select v-model="value" class="m-2" placeholder="Select" size="large">
  <el-option
    v-for="item in options"
    :key="item.value"
    :label="item.label"
    :value="item.value"
  />
</el-select>
  </div>

  <el-upload
  class="upload-demo"
  drag
  action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15"
  multiple
>
  <el-icon class="el-icon--upload"><upload-filled /></el-icon>
  <div class="el-upload__text">
    拖动文件至此或者 <em>点击上传文件</em>
  </div>
  <template #tip>
    <div class="el-upload__tip">
      jpg/png files with a size less than 500kb
    </div>
  </template>
</el-upload>

</div>

<el-row>
  <el-button @click="submit" type="primary"
    style="border: 1px solid #409EFF; border-radius: 4px; margin-top: 50px; margin-left: 48%; size:large">
    提交
  </el-button>
</el-row>
  `,

    data() {
      return {
        page: 0,
        value: null,
        options: []
      };
    },
    created() {
      axios.get('/page/php/taskname.php')
        .then(response => {
          this.options = response.data;
        })
        .catch(error => {
          console.error(error);
        });
    },
  methods: {
    pageChange(id) {
        this.page = id
    }
    , submit() {
      const upload = this.$refs.upload;
      const files = upload.uploadFiles;

      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file.raw);
      });

      axios.post('/page/php/download.php', formData)
        .then(response => {
          // 处理上传成功的响应
        })
        .catch(error => {
          // 处理上传失败的响应
        });
    }
  }

  }