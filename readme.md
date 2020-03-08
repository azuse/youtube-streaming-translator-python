# 实时翻译油管直播
> 开发中 v0.01   
> 测试开发于Python >= 3.6    

本脚本使用[谷歌云语音转文字api](https://cloud.google.com/speech-to-text)与[谷歌云翻译api](https://cloud.google.com/translate)实现对油管直播的实时字幕与翻译。  
脚本部分功能暂时不能使用代理，请配置全局透明代理或在不需要代理的网络环境下使用。
## 使用方法
0. 克隆本仓库
    ```bash
    git clone github.com/azuse/youtube-streaming-translator-python
    cd youtube-streaming-translator-python
    ```
1. 安装依赖库
    ```bash
    sudo apt-get install libsm6 libxrender1 libxext-dev
    pip install -r requirements.txt
    ```
    > 注：[安装pyaudio报错，需要自己编译安装portaudio：cannot-install-pyaudio-gcc-error](https://stackoverflow.com/questions/20023131/cannot-install-pyaudio-gcc-error)
2. 添加谷歌api key到Path中  
   
    Linux&Mac
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"
    ```
    Windows CMD
    ```bash
    set GOOGLE_APPLICATION_CREDENTIALS=[PATH]
    ```
    Windows PowerShell
    ```bash
    $env:GOOGLE_APPLICATION_CREDENTIALS="[PATH]"
    ```
    > 例： `export GOOGLE_APPLICATION_CREDENTIALS="/home/user/Downloads/service-account-file.json"`

3. 修改代理  
   * 不使用代理:
        请把代码中所有`response = requests.get(play.url, proxies=proxy)`的`, proxies=proxy`删掉
   * 使用代理:
        请把代码中所有`proxy = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}` 改为您使用的HTTP代理地址，**同时需要配置透明代理** （推荐[Clash的TAP模式](https://docs.cfw.lbyczf.com/contents/tap.html)）

4. 修改直播地址 
    
   将`url = "https://www.youtube.com/watch?v=8WwX_mlWHT0"`中的地址修改为您需要翻译的直播地址
   > 注：如果地址不是直播的话，程序会自动开始下载完整视频（应该会提示报错）
5. 启动脚本
   ```bash
   python main.google.py
   ```
   > 注：`main.ibm.py` 是过去测试的IBM Waston API版本，已废弃
6. 脚本输出  
   
   最后一行输出数字分别代表：  
   `开始下载的视频片段数 下载完成的视频片段数 下载失败的视频片段数 开始转码的视频片段数 转码完成已经发向谷歌API的片段数 从谷歌API收到回复（翻译完成）的片段数`

    > 注：目前脚本多进程退出不完善，可能导致进程变成孤儿进程，请注意

## 效果
![](res/pre.gif)

使用OBS加载浏览器效果(推荐自己优化CSS)
![](res/pre2.gif)

## 开发计划
* ~~使用Flask添加Web前端，方便OBS直接从浏览器将字幕加载到转播中~~优化CSS和JS
* 使用谷歌API实验性特性
* 解决代理问题
