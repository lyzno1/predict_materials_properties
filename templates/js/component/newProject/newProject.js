const ref = Vue.ref;

export default {
  template: `
  <div class="new1">
  <el-text class="mx-0">任务类型</el-text>
  <el-select v-model="value" class="m-2" placeholder="请选择任务类型" size="large">
    <el-option v-for="item in option" :key="item.label" :label="item.label" :value="item.value">
    </el-option>
  </el-select>
</div>

<div class="new1">
  <div class="block">
    <span class="demonstration">任务发布时间和截止时间</span>
    <el-date-picker v-model="time" type="datetimerange" range-separator="To" start-placeholder="Start date"
      end-placeholder="End date"></el-date-picker>
  </div>
</div>

<div class="new2">
  <el-row>
    <el-text class="mx-1">任务群体</el-text>
    <el-select v-model="people" class="m-2" placeholder="请选择任务群体" size="large">
      <el-option v-for="item in option1" :key="item.people" :label="item.label1" :value="item.people">

      </el-option>
    </el-select>
  </el-row>
</div>

<div class="new2">
  <el-row>
    <el-text class="mx-1">重复</el-text>
    <el-select v-model="repeat" class="m-2" placeholder="请选择发布周期" size="large">
      <el-option v-for="item in option4" :key="item.repeat" :label="item.label4"
        :value="item.repeat"></el-option>
    </el-select>
  </el-row>
</div>

<div>
  <div class="form-row">
    <el-text class="mx-2">任务题目</el-text>
    <el-input v-model="title" placeholder="请输入任务题目"></el-input>
  </div>
  
  <div class="form-row">
    <el-text class="mx-2">任务地点</el-text>
    <el-input v-model="path" placeholder="请输入任务地点"></el-input>
  </div>
  
  <div class="form-row">
    <el-text class="mx-2">任务内容</el-text>
    <div style="margin: 20px 0"></div>
    <el-input v-model="content" :autosize="{ minRows: 6, maxRows: 20 }" type="textarea" placeholder="请输入任务内容"
      style="width: 90%"></el-input>
  </div>
</div>

<el-row>
  <el-button @click="submit" type="primary"
    style="border: 1px solid #409EFF; border-radius: 4px; margin-left: 50%; size:large">
    提交
  </el-button>
</el-row>
    `,
  props: {
    userdata:Object
  },
  data() {
    return {
      value: '',
      time: null,
      people: null,
      repeat: null,
      title: '',
      path: '',
      content: '',

      option: [],
      option1: [],
      option4: [
        {
          repeat: '0',
          label4: '不重复',
        },
        {
          repeat: '1',
          label4: '每天',
        },
        {
          repeat: '7',
          label4: '每周',
        },
        {
          repeat: '30',
          label4: '每月',
        }
      ],
    }
  },
  mounted: function () {
    this.getMissonTemplates()
  },
  methods: {
    getMissonTemplates() {
      axios.post('/back/teacher/getMissonTemplates')
        .then((res) => {
          this.option = res.data['templates']
          this.option1 = res.data['misson']
        })
    },
    submit() {
      axios.post('/back/teacher/postMisson', {
        任务类型: this.value,
        开始日期: this.time[0],
        结束日期: this.time[1],
        任务群体: this.people,
        周期: this.repeat,
        任务名称: this.title,
        任务地点: this.path,
        任务描述: this.content,
        发布人姓名: this.userdata['姓名'],
        发布人工号: this.userdata['工号']
      }).then((res) => {
        if (res.data) {
          alert('发布成功');
          this.value = '';
          this.time = null;
          this.people = null;
          this.repeat = null;
          this.title = '';
          this.path = '';
          this.content = '';

          this.option = [];
          this.option1 = [];
        } else {
          alert('发布失败')
        }
      })

    },
  }
}

