import '../../path/xlsx.full.min.js'

export default {
    template: `
    <el-upload class="upload-demo" accept=".xls,.xlsx,.xlsm" :show-file-list="false" :auto-upload="false"
        :file-list="fileList" :limit=1 :on-change="handleChange" multiple>
        <el-button type="primary" text bg>
            <el-icon>
                <Plus />
            </el-icon>
            通过excle导入
        </el-button>
    </el-upload>
    `,
    emits: [`getDataList`],
    data() {
        return {
            fileList: [],
            file: "",
            dataList: []
        }
    },
    methods: {
        a(i) {
            this.$emit('getDataList', i)
        },
        handleChange(file, fileList) {
            this.fileList = [fileList[fileList.length - 1]]; // 只能上传一个Excel，重复上传会覆盖之前的        
            this.file = file.raw;
            let reader = new FileReader()
            let _this = this
            reader.readAsArrayBuffer(this.file)
            reader.onload = function () {
                let buffer = reader.result
                let bytes = new Uint8Array(buffer)
                let length = bytes.byteLength
                let binary = ''
                for (let i = 0; i < length; i++) {
                    binary += String.fromCharCode(bytes[i])
                }

                let wb = XLSX.read(binary, {
                    type: 'binary'
                })
                let outdata = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]])
                _this.file = ''
                _this.fileList = []
                _this.$emit('getDataList', outdata)
            }
        },

    }
}