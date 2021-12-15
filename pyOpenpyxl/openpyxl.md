# openpyxl
[URL](https://openpyxl.readthedocs.io/en/stable/)
## 准备
```python
from openpyxl import Workbook, load_workbook
```
## WorkBook
```python
wb = Workbook()   # 新建 Workbook
wb = load_workbook("Excel.xlsx",data_only=True) # 读取文件 

ws = wb.active    # 选择当前Worksheet
ws = wb['sheet']  # 选择sheet工作表

wb.worksheets     # 返回Worksheet对象列表
wb.sheetnames     # 返回Worksheet名列表
wb.index(ws)      # 返回Worksheet的index

ws = wb.create_sheet('sheetname',[index])  # 创建新Worksheet 
wb.copy_worksheet(ws)   # 复制Worksheet 为 {name} Copy
wb.remove(ws)     # 删除Worksheet
wb.move_sheet(ws,offset)     # 移动Worksheet offset偏移量

wb.save('Excel.xlsx')  # 保存
wb.close()             # 关闭
```
## WorkSheet
### Read
> Cell内容是函数时，读取的内容是函数。
> 想要数据时，需要读取Excel时设置data_only=True
```python
# Access A Cell
cell = ws["A1"].value
cell = ws.cell(row=1,column=1)

cell.coordinate     # 返回Cell 坐标
cell.row            # 返回Cell 所在 row
cell.column         # 返回Cell 所在 col的index
cell.column_letter  # 返回Cell 所在 col名

ws.max_row          # 返回工作表最大row
ws.max_column       # 返回工作表最大column

# Access many Cells
rows = ws[1:2]
cols = ws["A:B"]

ws.rows / ws.columns / ws.values # 返回cell对象 生成器 

for row in rows:
    for cell in row:
        print(cell)

# Use iter_rows/iter_cols
for row in ws.iter_rows(min_row=1, max_col=3, max_row=2, values_only=True):
    print(row)
    for cell in row:
        print(cell)
```
### Write
```python
# Write A Cell
ws.cell(row,col,value)
ws["A1"] = "value"

# Write A Row
ws.append([1,2,3])  # 在最后一行添加数据 

# Write formulae
ws["A1"] = "=SUM(B1+1)"

# Write Range
for row in ws.iter_rows(min_row=1, max_col=3, max_row=2):
    for cell in row:
        cell.value = "value"


# Write date
import datetime

ws["A2"] = datetime.datetime(2021,12,15)
ws["A2"].number_format   # 返回 'yyyy-mm-dd h:mm:ss'
```

### Setup
```python
# title
ws.title = "Sheet"  # 设置工作表名
ws.sheet_properties.tabColor = "1072BA"   # 设置工作表颜色
```
#### Cell type
```python
ws["A1"].data_type = "n"  # 默认为 n 数值  s 字符串  d 日期时间
```
#### Insert / Delete rows / cols
```python
ws.insert_rows(5,1)  # index amount  默认为1

ws.delete_cols(6.3)
```
#### Move range
```python
ws.move_range("B1:C3",rows=-1,cols=2,translate=True)
# translate 为True时 函数参考也会改变
```

#### Merge / Unmerge cells
```python
ws.merge_cells("A2:D3")
ws.unmerge_cells("A2:D3")

ws.merged_cell_ranges    # 返回已经合并的cell 列表

ws.merge_cells(start_row=2,start_column=1,end_row=4,end_col=4)
ws.unmerge_cells(start_row=2,start_column=1,end_row=4,end_col=4)
```
#### Group
```python
ws.column_dimensions.group('A','D', hidden=True)    # 隐藏 True
ws.row_dimensions.group(1,10, hidden=True)
```
#### filter
```python
ws.auto_filter.ref = "A:B"
```
#### Insert image
```python
from openpyxl.drawing.image import Image
img = Image("pic.png")
ws.add_image(img,"A1")
```
## Chart
```python
from openpyxl.chart import Reference
Reference(min_row=1,min_col=1,max_row=10,max_col=10)

ws.add_chart(chart,"A1")
```
### Area Chart
```python
from openpyxl.chart import AreaChart,AreaChart3D,Reference,Series,legend

Chart = AreaChart()
Chart.width = 20           # 表大小
Chart.height = 13         
Chart.title = "Chart"      # 设置表名
Chart.style = 10           # 设置风格
Chart.x_axis.title = "X"   # 设置X轴名
Chart.y_axis.title = "Y"   # 设置Y轴名
Chart.y_axis.majorGridlines = None   # 取消网格
Chart.x_axis.scaling.min = -10       # 轴最小值
Chart.x_axis.scaling.max = 10        # 轴最大值
Chart.x_axis.majorUnit = 1           # 轴步长
Chart.x_axis.minorTickMark = "in"    # 刻度线显示位置 "out"

Chart.legend = legend.Legend("tr")  # b 底部 tr 右上 l 左　r 右 t 上　None 不显示 

Chart.add_data(data,titles_from_data=True)
Chart.set_categories(cats)
```
### Bar and Column Chart
```python
from openpyxl.chart import BarChart,BarChart3D,Series,Reference
Chart = BarChart()
Chart.type = "col"/"bar"    # col 纵向 bar 横向

Chart.grouping = "stacked"  # {'percentStacked', 'standard', 'clustered', 'stacked'}
Chart.overlap = 100         # 積み上げ図
```
### Line Chart
```python
from openpyxl import Workbook
from openpyxl.chart import (
    LineChart,
    Reference,
)
from openpyxl.chart.axis import DateAxis

Chart = LineChart()

Chart.grouping = "stacked"        # 積み上げ

s1 = Chart.series[0]
s1.marker.symbol = "triangle"     # 三角形标记
s1.marker.graphicalProperties.solidFill = "FF0000" # 标记填充颜色
s1.marker.graphicalProperties.line.solidFill = "FF0000" # Marker outline

s1.graphicalProperties.line.solidFill = "00AAAA"   # 线颜色
s1.graphicalProperties.line.dashStyle = "sysDot"   # 线的形状
s1.graphicalProperties.line.width = 100050         # 线的尺寸

Chart.x_axis = DateAxis(crossAx=100)    # 设置坐标轴为时间样式
Chart.x_axis.number_format = 'd-mmm'
Chart.x_axis.majorTimeUnit = "days"
```
### Scatter Chart
```python
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)
Chart = ScatterChart()
for i in range(2, 4):
    values = Reference(ws, min_col=i, min_row=1, max_row=7)
    series = Series(values, xvalues, title_from_data=True)
    Chart.series.append(series)
```
## Style
```python
from openpyxl.styles import PatternFill,Border,Side,Alignment,Protection,Font,Color

from openpyxl.styles import NamedStyle

Mystyle = NamedStyle(name="style_name",
                     font=Font(),
                     fill=PatternFill(),
                     border=Border(),
                     alignment=Alignment()
                     )
wb.add_named_style(Mystyle)
cell.style = Mystyle
cell.style = "style_name"

```
### Color
```python
Color(index=0)
Color(rgb="000000")
```
### Font
```python
cell.font = Font
font = Font(name='Calibri',
            size=11,
            bold=False,           # 加粗
            italic=False,         # 斜体
            vertAlign=None,       # 垂直对齐
            underline='none',
            strike=False,         
            color='FF000000')
```
### Fill
```python
cell.fill = PatternFill()
PatternFill(patternType='solid',fgColor=Color(),bgColor=Color())
# fg 前景色
# bg 后景色  设置反应是黑色 
```
### Border
```python
cell.border = Border()

Side(style='thin',color=Color(index=0))
# thin medium thick 粗 dashed  虚线  dotted  点线 double 
Border(left=Side(),right=Side(),top=Side(),bottom=Side())
```
### Alignment
```python
cell.alignment = Alignment()
Alignment(horizontal="fill",vertical="center")
```

