# Selenium
## ׼��
```python
from selenium.webdriver import Chrome
```
> ����Chrome�汾����Ҫ����chromedriver
## �������
```python
from selenium.webdriver import Chrome
web = Chrome()
web.get('https://www.baidu.com')

web.page_source  # ��ȡ��Ⱦ���ҳ�����
```
## ����
```python
from selenium.webdriver.chrome.options import Options

opt = Options()
opt.add_argument("--headless")      # ��������� ��̨����
opt.add_argument("--disable-gpu")   # ����GPU
opt.add_argument('--disable-blink-features=AutomationControlled')  # �����Զ���ʶ��Ϊ��
web = Chrom(opt=opt)
```
## Ѱ��Ŀ��
```python
web.find_element_by_xpath('Xpath')            # Ѱ��dangeĿ��
web.find_elements_by_xpath('Xpath')           # Ѱ�Ҷ������Xpath��Ŀ��
web.find_element_by_class_name('class_name')  # ����classNameѰ��Ŀ��
web.find_element_by_id('id')                  # ����idѰ��Ŀ��
```
## ����
```python
from selenium.webdriver.common.keys import Keys
target = web.find_element_by_xpath('Xpath')

target.click()                            # ���
target.text                               # ��Ź�ı�
target.send_keys('string',Keys.ENTER)     # �����ı�
target.screemshot_as_png                  # ��ȡͼ��
```
### �л�����
```python
# �´���Ĭ�ϲ����л�
web.switch_to.window(web.window_handles[0])

iframe = web.find_element_by_xpath('xpath')
web.switch_to.frame(iframe)    # �л����Ӵ���

web.close()  # �رյ�ǰ����  �����л�
```
### �����˵�
```python
from selenium.webdriver.support.select import Select

target = web.find_element_by_xpath('Xpath')
selecter = Select(target)

for i in range(len(selecter.options)):     # i ѡ������λ�� 
    selecter.select_by_index(i)
    selecter.select_by_value('value')      # <option value='value'>
    selecter.select_by_visible_text('txt') # �ı�
```
### һ�����Ĳ���
```python
from selenium.webdriver.common.action_chains import ActionChains

ActionChains(web).move_to_element_with_offset(target, x, y).click().perform() 
# ƫ��x y ���

ActionChains(web).drag_and_drop_by_offset(target,100,0).perform() # �����ק
# x��ƫ���� 100 y��ƫ���� 0  ��Ҫperform ��ִ��
```