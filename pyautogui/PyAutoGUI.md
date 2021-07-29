# PYAUTOGUI

```python
pyautogui.size()           # 返回屏幕宽高 
pyautogui.PAUSE = 1        # 动作等待间隔
pyautogui.FAILSAFE = True  # 启用左上角停止
pyautogui.sleep()          # 等待
```

## 鼠标
```python
pyautogui.position()       # 返回鼠标当前位置
```

### 移动
```python
pyautogui.moveTo(x,y,t)    # 移动到坐标x,y
pyautogui.moveRel(x,y,t)   # 移动偏移量x,y
```
### 点击
```python
pyautogui.mouseDown()           # 按住
pyautogui.mouseUp()             # 松开
pyautogui.click([x,y,button])   # 在(x,y)处点击
pyautogui.doubleClick()         # 双击 
pyautogui.rightClick()          # 鼠标右键点击
pyautogui.middleClick()         # 鼠标中键点击
```
### 拖动
```python
pyautogui.dragTo()     # 点击拖动到(x,y)处
pyautogui.dragRel()    # 点击拖动偏移量(x,y)
```

### 滚动
```python 
pyautogui.scroll()
```

## 键盘

```python
pyautogui.typewrite()　　　# 可以传递列表
pyautogui.keyDown()       # 按下 
pyautogui.keyUp()         # 松开
pyautogui.press()         # 按下加松开
pyautogui.hotkey()        # 快捷键

```