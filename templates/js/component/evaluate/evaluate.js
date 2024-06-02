export default {
    template: `
    <!--<el-container id="main-container">
        <el-aside id="main-container-aside">
            <div v-for="project in projects" :key='project.id'>
                <el-button @click="projectId=project.id" :class="{projectId: project.id==projectId}" text>
                    任务{{project.data}}
                </el-button>
            </div>
        </el-aside>

        <el-main id="main-container-main">
            <div v-if="projectId!==0 && studentFiles.length>0">
                <el-table :data="studentFiles" border stripe style="width: 100%">
                    <el-table-column prop="applicant" label="申请人" width="100" />
                    <el-table-column prop="id" label="学号" width="110" />
                    <el-table-column prop="academy" label="学院" width="120" />
                    <el-table-column prop="schedule" label="状态" />
                    <el-table-column label="操作" width="90">
                    <template #default="scope">
                        <el-button type="primary" @click="evaluate=true;evaluateOpen(scope.row.id)">评价</el-button>         
                    </template>
                    </el-table-column>
                </el-table>
            </div>
        </el-main>-->
        <el-container id="main-container">
        <el-aside id="main-container-aside">
            <div v-for="(p,id) in projects" :key='id+1'>
                <el-button @click="projectId=id+1;project=p;getStudentAns(p.任务编号)" :class="{projectId: id==projectId}"
                    text>
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

                <el-table v-if='projectId!==0 && missionStudent.length>0' :data="missionStudent" border stripe
                    style="width: 100%;margin-top: 20px;">
                    <el-table-column prop="姓名" label="申请人" width="100"></el-table-column>
                    <el-table-column prop="学号" label="学号" width="110"></el-table-column>
                    <el-table-column prop="学院" label="学院" width="120"></el-table-column>
                    <el-table-column prop="班级" label="班级"></el-table-column>
                    <el-table-column label="操作" width="160">
                        <template v-slot="scope">
                            <el-button type="primary"
                                @click="evaluate=true;getStudentFiles(scope.row.学号,project.任务编号)">评价</el-button>
                        </template>
                    </el-table-column>

                </el-table>
            </div>
        </el-main>
    </el-container>

    <el-dialog v-model="evaluate" title="Shipping address">

        <el-form :model="evaluateData">
            <el-form-item label="工时:">
                <el-input v-model="evaluateData[0].工时" autocomplete="off" />
            </el-form-item>
            <el-form-item label="评分:">
                <el-input v-model="evaluateData[0].分数" autocomplete="off" />
            </el-form-item>
            <el-form-item label="等级评价:">
                <el-radio-group v-model="evaluateData[0].分数">
                    <el-radio label=100>优秀</el-radio>
                    <el-radio label=80>良好</el-radio>
                    <el-radio label=60>及格</el-radio>
                    <el-radio label=40>不及格</el-radio>
                    <el-radio label=0>零分</el-radio>
                </el-radio-group>
            </el-form-item>
            <el-form-item label="文字评价:">
                <el-input v-model="evaluateData[0].评价" :rows="4" type="textarea" placeholder="Please input">
                </el-input>
            </el-form-item>
        </el-form>

        <template #footer>
            <span class="dialog-footer">
                <el-button @click="evaluate = false">取消</el-button>
                <el-button type="primary" @click="evaluateUpData()">
                    确定
                </el-button>
            </span>
        </template>

        <h1 v-if="studentFiles==[]">该同学尚未提交课程成果</h1>
        <el-table v-if="studentFiles!=[]" height="calc(200px)" stripe border :data="studentFiles" style=" margin-top:20px;width: 100%"
            :cell-style="{ textAlign: 'center' }" :header-cell-style="{ 'text-align': 'center' }">
            <el-table-column prop="文件名" label="文件名"></el-table-column>
            <el-table-column prop="时间" label="提交时间"></el-table-column>
            <el-table-column label="操作">
                <template v-slot="scope" style="display: flex; flex-dicrection: column;">
                    <el-button type="primary" plain @click=" downloadFile(scope.row.地址)">下载</el-button>
                </template>
            </el-table-column>
        </el-table>

    </el-dialog>
    `,
    props: {
        userdata: Object
    },
    data() {
        return {
            projects: null,
            projectId: 0,
            evaluate: false,

            tempStudent: '',
            tempMission: '',
            evaluatePresent: false,

            missionStudent: [],
            studentFiles: [],
            evaluateData: [{
                '工时': 0,
                '分数 ': 0,
                '评价': ''
            }]
        }
    },
    mounted: function () {
        console.log(this.evaluateData);
        this.getProjectList();
    },
    methods: {
        getProjectList() {
            axios.post('/back/teacher/getProject', {
                发布人: this.userdata['工号']
            })
                .then((res) => {
                    this.projects = res.data
                })
        },

        getStudentAns(missionId) {
            axios.post('/back/teacher/getStudnetAns', {
                任务编号: missionId
            }).then((res) => {
                this.missionStudent = res.data;
            })
        },

        getStudentFiles(No, missionId) {
            this.tempMission = missionId
            this.tempStudent = No
            axios.post('/back/teacher/getStudentFiles', {
                学号: No,
                任务编号: missionId
            }).then((res) => {
                this.studentFiles = res.data.files;
                if (res.data.evaluatie.length > 0) {
                    console.log(res.data.evaluatie);
                    this.evaluateData = res.data.evaluatie;
                    this.evaluatePresent = true
                }
            })
        },

        downloadFile(url) {
            window.open(url, '_self')
        },

        evaluateUpData() {
            axios.post('/back/teacher/upEvaluate', {
                mission: this.tempMission,
                No: this.tempStudent,
                工时: this.evaluateData[0].工时,
                分数: this.evaluateData[0].分数,
                评价: this.evaluateData[0].评价,
                present: this.evaluatePresent
            })
                .then((res) => {
                    if (res.data) {
                        this.$notify.success({
                            message: '评价成功',
                        })
                        this.evaluatePresent = false
                        this.evaluateData = [{
                            '工时': 0,
                            '分数': 0,
                            '评价': ''
                        }]
                        this.evaluate = false
                    }
                    else {
                        this.$notify.error({
                            message: '评价失败',
                        })
                    }
                })


        }
    }

}