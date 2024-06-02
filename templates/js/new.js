

// 当用户点击弹窗以外的区域时关闭弹窗
window.onclick = function(event) {
    if (event.target == modal) {
        closeModal();
    }
}
var previousFocusedElement; // 全局变量用于记录之前的焦点元素

// 打开弹窗的函数
function openModal() {

    var modal = document.getElementById('periodic-table-modal');
    modal.style.display = 'block';

    // 获取当前焦点元素
    var activeInput = document.activeElement;

    // 如果没有焦点或者焦点不在输入框上，则默认将焦点设置到第一个输入框上
    if (!activeInput || (activeInput.tagName !== 'INPUT' && activeInput.tagName !== 'TEXTAREA')) {
        document.getElementById('atom1').focus();
    }


}

// 关闭弹窗的函数
function closeModal() {
    var modal = document.getElementById('periodic-table-modal');
    modal.classList.add('hidden'); // 添加 'hidden' 类，触发渐变关闭效果
    setTimeout(() => {
        modal.style.display = 'none'; // 在渐变结束后隐藏弹窗
        modal.classList.remove('hidden'); // 移除 'hidden' 类
    }, 500); // 等待 0.5 秒后隐藏弹窗

}

// 点击元素处理函数
function handleSymbolClick(symbol) {
    console.log("Clicked symbol:", symbol); // 确保正确获取到符号

    var activeInput = document.activeElement; // 获取当前焦点元素
    if (document.getElementById('atom1').value.length > 0 && document.getElementById('atom2').value.length > 0) {
        document.getElementById('atom1').value = ''; // 清空文本框1的值
        document.getElementById('atom2').value = ''; // 清空文本框2的值
        document.getElementById('atom1').focus(); // 聚焦到文本框1
    }

    
    if (document.getElementById('atom1').value.length > 0) {
        activeInput = document.getElementById('atom2');
    } else if (document.getElementById('atom1').value.length == 0) {
        activeInput = document.getElementById('atom1');
    } 
    
    console.log("Active Input Tag Name:", activeInput.tagName);

    // 根据焦点元素来确定要填入的文本框
    if (activeInput.id === 'atom1') {
        activeInput.value += symbol; // 将符号填入当前焦点的文本框
        console.log("Adding symbol to atom1Input:", symbol);
        console.log("Atom1 Input Value:", activeInput.value); // 打印当前焦点文本框的值
    } else if (activeInput.id === 'atom2') {
        activeInput.value += symbol; // 将符号填入当前焦点的文本框
        console.log("Adding symbol to atom2Input:", symbol);
        console.log("Atom2 Input Value:", activeInput.value); // 打印当前焦点文本框的值
    }

    closeModal(); // 关闭弹窗
    console.log(document.getElementById('atom1').value)
    console.log(document.getElementById('atom2').value)
}



function handleElementClick(element) {
    const symbol = element.querySelector('.symbol').textContent;
    handleSymbolClick(symbol); // 将符号传递给 handleSymbolClick 函数
}

function predictDistance() {
    const atom1 = document.getElementById('atom1').value;
    const atom2 = document.getElementById('atom2').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ atom1, atom2 }),
    })
    .then(response => response.json())
    .then(data => {
        alert(`预测的原子间距离为: ${data.prediction} Å`);
    })
    .catch((error) => {
        console.error('Error:', error);
        showErrorAlert('发生错误，请稍后再试');
    });
    document.getElementById('atom1').value = ''; // 清空文本框1的值
    document.getElementById('atom2').value = ''; // 清空文本框2的值
}


function showErrorAlert(message) {
    document.getElementById('alertMsg').innerText = message;
    document.getElementById('alertBox').style.display = 'block';
}