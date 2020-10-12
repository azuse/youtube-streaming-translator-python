# 实时翻译油管直播
> 测试开发于Python >= 3.6    

本脚本使用[谷歌云语音转文字api](https://cloud.google.com/speech-to-text)与[谷歌云翻译api](https://cloud.google.com/translate)/[百度翻译](https://api.fanyi.baidu.com/)实现对油管直播的实时字幕与翻译。  

## 使用方法
### 0. 克隆本仓库
   
 ```bash
 git clone github.com/azuse/youtube-streaming-translator-python
 cd youtube-streaming-translator-python
 ```

### 1. 安装依赖库
   
 ```bash
 pip install -r requirements.txt
 ```
 > 注1：从[noise_reduction](https://github.com/dodiku/noise_reduction)中复制了降噪功能，降噪功能需要安装[SOX](https://sourceforge.net/projects/sox/files/sox/14.4.2/)，不需要使用可以把代码直接注释掉。

### 2. 添加谷歌api key到Path中  
   
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

### 3. *使用百度翻译API
   
脚本可以选择使用百度翻译API或谷歌翻译API，使用百度翻译API需要：
 * 把翻译函数中的`baidu`设置为`baidu=True`（目前在245行）
 * 在`baiduapi.py`中设置自己的`APP ID`和`SECRET KEY`


### 4. 修改代理  
   * 不使用代理:
        直接运行
   * 使用代理:  
        Ubuntu/Linux
        ```bash
        export http_proxy=http://127.0.0.1:7890
        export https_proxy=http://127.0.0.1:7890
        ```
        Windows
        ```cmd
        set http_proxy=http://127.0.0.0.1:7890
        set https_proxy=http://127.0.0.0.1:7890
        ```

### 5. 修改直播地址 
    
将`url = "https://www.youtube.com/watch?v=ylFDswiFduE"`中的地址修改为您需要翻译的直播地址
> 注：如果地址不是直播的话，程序会自动开始下载完整视频（应该会提示报错）

### 6. 启动脚本
 
```bash
python main.google.py
```

### 7. 脚本输出  
   
最后一行输出数字分别代表：  
`开始下载的视频片段数 下载完成的视频片段数 下载失败的视频片段数 开始转码的视频片段数 转码完成已经发向谷歌API的片段数 从谷歌API收到回复（翻译完成）的片段数`

### 8. 浏览器输出
   
脚本会自动在0.0.0.0:5000上开一个http服务器，同时在0.0.0.0:5001和0.0.0.0:5002上开socket服务器。
访问`http://127.0.0.1:5000`或`http://你的服务器ip:5000`会打开显示字幕用的网页，网页分别从5001和5002的socket读取日文与中文字幕。
网页的CSS有待优化，目前只是一个测试效果。

### 9.  *ChunkSize
     
油管的直播是以1s左右时间一段.ts文件的格式传输的，在将视频段转为音频后，可以决定要把多少段视频合在一起送给谷歌的流式音频识别API，决定多少段合并在一起的变量叫`chunksize`。  

如果设置为较小的值（1），每次音频识别API可能只会返回一个词，日译中效果不好(<-语音识别和翻译的ChunkSize也许可以分开设置？)；如果设置为较大值（10），意味着每10s才会更新一次字幕；  

所以`chunksize`的大小对识别效果有很大影响，目前感觉设置为`4`效果较好，每4s一个片段。但是，如果一句话跨越两个4s片段，流式音频识别API对于两个连起来的片段很多时候还是会拆成两句话来识别，导致中间内容丢失。  

## 效果
![](res/pre.gif)

使用OBS加载浏览器效果(推荐自己优化CSS)
![](res/pre3.gif)
![](res/pre2.gif)

## 开发计划
* ~~使用Flask添加Web前端，方便OBS直接从浏览器将字幕加载到转播中~~
* 优化前端CSS和JS
* 调教语音转文字和翻译
* ~~解决代理问题~~
* ~~增加百度翻译api~~
* 增加彩云小译API 增加IBM watson API

