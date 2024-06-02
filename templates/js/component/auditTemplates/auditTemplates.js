export default {
    template: `
    <el-container id="main-container">
            <el-aside id="main-container-aside">
              <el-button @click="addAuditTemplates()">
                <el-icon>
                  <Plus />
                </el-icon>
                添加审核流程
              </el-button>
              <div v-for="(data,i) in templates" :key='i'>
                <el-button @click="templatesId=1,tempTemplates=data">
                  {{data.模板名称}}
                </el-button>
              </div>
            </el-aside>

            <el-main id="main-container-main">

              <div v-if="templatesId==0">
                <el-form :model="tempTemplates">
                  <el-form-item label="模板名称">
                    <el-input v-model="tempTemplates.模板名称"></el-input>
                  </el-form-item>

                  <div v-for="(item, i) in tempTemplates.模板流程" class="templatesCard">
                    <el-card>
                      <el-row class="audit">
                        <span>流程{{i+1}} </span>
                        <el-tooltip content="需要通过人数，默认为0，及需要全部通过" placement="top">
                          <el-select v-model="item.控制" style="width: 70px;padding-right: 15px;">
                            <el-option v-for="item of item.审核人员.length+1" :label="item-1" :value="item-1" />
                          </el-select>
                        </el-tooltip>
                        <div v-for="i in item.审核人员">
                          <el-cascader v-model="i.人员" :options="templatesPeople" :show-all-levels="false"
                            clearable></el-cascader>
                        </div>
                        <el-button style="padding-left: 20px;" @click="addPeople(item.审核人员)">
                          <el-icon>
                            <Plus />
                          </el-icon>
                        </el-button>
                      </el-row>
                    </el-card>
                  </div>

                  <el-row>
                    <el-col :span="12">
                      <el-button @click="addTemplates()" style="width: 80%;">添加流程</el-button>
                    </el-col>
                    <el-col :span="12">
                      <el-button @click="upTemplates() " style="width: 80%;">上传模板</el-button>
                    </el-col>

                  </el-row>
                </el-form>
              </div>

              <div v-if=" templatesId!==0">
                <el-form :model="tempTemplates">
                  <el-form-item label="模板名称">
                    <el-input v-model="tempTemplates.模板名称"></el-input>
                  </el-form-item>

                  <div v-for="(item, i) in tempTemplates.模板流程" class="templatesCard">
                    <el-card>
                      <el-row class="audit">
                        <span>流程{{i+1}} </span>
                        <el-tooltip content="需要通过人数，默认为0，及需要全部通过" placement="top">
                          <el-select v-model="item.控制"  style="width: 70px;padding-right: 15px;">
                            <el-option v-for="item of item.审核人员.length+1" :label="item-1" :value="item-1" />
                          </el-select>
                        </el-tooltip>
                        <div v-for="i in item.审核人员">
                          <el-cascader v-model="i.人员" :placeholder="i.人员" :options="templatesPeople" :show-all-levels="false"
                            clearable></el-cascader>
                        </div>
                        <el-button style="padding-left: 20px;" @click="addPeople(item.审核人员)">
                          <el-icon>
                            <Plus />
                          </el-icon>
                        </el-button>
                      </el-row>
                    </el-card>
                  </div>

                  <el-row>
                    <el-col :span="8">
                      <el-button @click="addTemplates()" style="width: 80%;">添加流程</el-button>
                    </el-col>
                    <el-col :span="8">
                      <el-button @click="upTemplates() " style="width: 80%;">上传模板</el-button>
                    </el-col>
                    
                    <el-col :span="8">
                    <el-popconfirm confirm-button-text="是" cancel-button-text="否" title="是否删除本模板?"
                    @confirm="del(tempTemplates.模板名称)">
                    <template #reference>
                      <el-button type="danger"  style="width: 80%;">删除模板</el-button>
                    </template>
                    </el-popconfirm>
                    </el-col>

                  </el-row>
                </el-form>
              </div>
            </el-main>
          </el-container>
    `,
    data() {
        return {
            templatesId: 0,
            projects: null,
            templatesPeopleList: [],
            templatesPeople: [
                {
                    label: '一级审批',
                    children: [{
                        label: '全部一级审批',
                        value: 'all 1'
                    }]
                },
                {
                    label: '二级审批',
                    children: [{
                        label: '全部二级审批',
                        value: 'all 2'
                    }]
                },
                {
                    label: '三级审批',
                    children: [{
                        label: '全部三级审批',
                        value: 'all 3'
                    }]
                },
            ],
            tempTemplates: {
                模板名称: '',
                模板流程: [
                    {
                        控制: 0,
                        审核人员: [{
                            人员: []
                        }]
                    },
                ],
            },
            templates: null
        }
    },
    mounted: function () {
        this.getTemplates();
        this.getTemplatesPeople();
    },
    methods: {
        getTemplates() {
            axios.post('/back/admin/auditTemplates/getAuditTemplates')
                .then((res) => {
                    this.templates = res.data;
                })
        },
        getTemplatesPeople() {
            axios.post('/back/admin/auditTemplates/getAuditPeople')
                .then((res) => {
                    this.templatesPeopleList = res.data
                    let temp = res.data
                    for (let i = 0; i < temp[0].length; i++) {
                        this.templatesPeople[0].children.push({
                            label: temp[0][i].姓名,
                            value: [temp[0][i].姓名, temp[0][i].工号]
                        })
                    }
                    for (let i = 0; i < temp[1].length; i++) {
                        this.templatesPeople[1].children.push({
                            label: temp[1][i].姓名,
                            value: [temp[1][i].姓名, temp[1][i].工号]
                        })
                    }
                })
        },
        addAuditTemplates() {
            this.templatesId = 0
            this.tempTemplates = {
                模板名称: '',
                模板流程: [
                    {
                        控制: 0,
                        审核人员: [{
                            人员: []
                        }]
                    },
                ],
            }
        },
        addTemplates() {
            this.tempTemplates.模板流程.push({
                控制: 0,
                审核人员: [{
                    人员: []
                }]
            })
        },
        addPeople(i) {
            i.push({
                人员: []
            })
        },
        //上传模板
        upTemplates() {
            axios.post('/back/admin/auditTemplates/upTemplates', {
                templates: this.tempTemplates,
                people: this.templatesPeopleList
            }).then((res) => {
                if (res.data) {
                    alert('添加成功')
                    this.tempTemplates = {
                        模板名称: '',
                        模板流程: [
                            {
                                控制: 0,
                                审核人员: [{
                                    人员: ''
                                }]
                            },
                        ]
                    }
                    this.getTemplates()
                } else {
                    alert('添加失败')
                }
            })
        },
        del(name) {
            console.log(name);
            axios.post('/back/admin/auditTemplates/delAuditTemplates', {
                模板名称: name,
            }).then((res) => {
                if (res.data) {
                    alert('删除成功')
                    this.tempTemplates = {
                        模板名称: '',
                        模板流程: [
                            {
                                控制: 0,
                                审核人员: [{
                                    人员: ''
                                }]
                            },
                        ],
                    }
                    this.getTemplates()
                } else {
                    alert('删除失败')
                }
            })
        }
    }
}