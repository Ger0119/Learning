# Selenium
## 准备
```python
from selenium.webdriver import Chrome
```
> 根据Chrome版本，需要下载chromedriver
## 打开浏览器
```python
from selenium.webdriver import Chrome
web = Chrome()
web.get('https://www.baidu.com')

web.page_source  # 获取渲染后的页面代码
```
## 设置
```python
from selenium.webdriver.chrome.options import Options

opt = Options()
opt.add_argument("--headless")      # 不打开浏览器 后台运行
opt.add_argument("--disable-gpu")   # 禁用GPU
opt.add_argument('--disable-blink-features=AutomationControlled')  # 设置自动化识别为否
web = Chrom(opt=opt)
```
## 寻找目标
```python
web.find_element_by_xpath('Xpath')            # 寻找dange目标
web.find_elements_by_xpath('Xpath')           # 寻找多个符合Xpath的目标
web.find_element_by_class_name('class_name')  # 根据className寻找目标
web.find_element_by_id('id')                  # 根据id寻找目标
```
## 操作
```python
from selenium.webdriver.common.keys import Keys
target = web.find_element_by_xpath('Xpath')

target.click()                            # 点击
target.text                               # 互殴文本
target.send_keys('string',Keys.ENTER)     # 输入文本
target.screemshot_as_png                  # 获取图像
```
### 切换窗口
```python
# 新窗口默认不被切换
web.switch_to.window(web.window_handles[0])

iframe = web.find_element_by_xpath('xpath')
web.switch_to.frame(iframe)    # 切换至子窗口

web.close()  # 关闭当前窗口  不被切换
```
### 下拉菜单
```python
from selenium.webdriver.support.select import Select

target = web.find_element_by_xpath('Xpath')
selecter = Select(target)

for i in range(len(selecter.options)):     # i 选项索引位置 
    selecter.select_by_index(i)
    selecter.select_by_value('value')      # <option value='value'>
    selecter.select_by_visible_text('txt') # 文本
```
### 一连串的操作
```python
from selenium.webdriver.common.action_chains import ActionChains

ActionChains(web).move_to_element_with_offset(target, x, y).click().perform() 
# 偏移x y 点击

ActionChains(web).drag_and_drop_by_offset(target,100,0).perform() # 点击拖拽
# x轴偏移量 100 y轴偏移量 0  需要perform 来执行
```