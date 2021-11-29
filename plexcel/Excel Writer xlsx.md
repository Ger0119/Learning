# Excel::Writer::XLSX

## 准备
```perl
use Excel::Writer::XLSX;
$workbook = Excel::Writer::XLSX->new("excel.xlsx");  #创建文件
$workbook->close;  #关闭

use Excel::Writer::XLSX::Utility;
( $row, $col ) = xl_cell_to_rowcol( 'C2' );    # (1, 2)
$str           = xl_rowcol_to_cell( 1, 2 );    # C2
```
## sheet
```perl
$worksheet = $workbook->add_worksheet("Sheet");               # Sheet
```
### Write
```perl
$worksheet->write( $row, $column, $token, $format );
$worksheet->write(1,2,"P1");   #row column  Index 0开始
$worksheet->write("D4","P2");
```
#### write_number
```perl
$worksheet->write("A1","0001000");　　　　　　# 1000
$worksheet->write_number("B1","0001000");   # 1000
```

#### write_string
```perl
$worksheet->write_string("C1","0001000");   # 0001000
$red  = $workbook->add_format( color => 'red' );   $format
$blue = $workbook->add_format( color => 'blue' );  $format
$worksheet->write_rich_string("A1",     #复杂格式字符串
		'This is ', $red, 'red', ' and this is ', $blue, 'blue' );
```

#### write_formula
```perl
$worksheet->write("A1","=SUM(b:b)");        # same as write_formula()
$worksheet->write_formula("A2","=if(A1>0,1,0)");
$worksheet->write_array_formula('A1:A3', '=SUM(B1:C1)'); # 返回结果
```
#### write_row write_col
```perl
$worksheet->write("A1",[\@lst]);   #row方向    same as write_col
$worksheet->write("C1",\@lst);     #column方向 same as write_row
$worksheet->write_col("A1",\@lst);
$worksheet->write_row("C1",\@lst);
```
#### write_boolean
```perl
$worksheet->write_boolean( 'A1', 1          );  # TRUE
$worksheet->write_boolean( 'A2', 0          );  # FALSE
$worksheet->write_boolean( 'A3', undef      );  # FALSE
```
#### write_blank
```perl
$worksheet->write("A1");      # 空值时不能覆写  undef "" 等
$worksheet->write_blank("A1"," ")  # 可以覆写
```
#### write_date_time
```perl
$date_format = $workbook->add_format(num_format =>'yyyy-mm-ddThh:mm:ss.sss' );
$worksheet->write_date_time( 'A1', '2004-05-13T23:20', $date_format );
```
### set
#### set_selection( $first_row, $first_col, $last_row, $last_col )
```perl
$worksheet1->set_selection( 3, 3 );          # 1. Cell D4. 
$worksheet2->set_selection( 3, 3, 6, 6 );    # 2. Cells D4 to G7.
$worksheet3->set_selection( 6, 6, 3, 3 );    # 3. Cells G7 to D4.
$worksheet4->set_selection( 'D4' );          # Same as 1.
$worksheet5->set_selection( 'D4:G7' );       # Same as 2.
$worksheet6->set_selection( 'G7:D4' );       # Same as 3.
```
#### set_row
> $row, $height, $format, $hidden, $level, $collapsed )
> $hidden 隐藏  $level  グループ化
```perl
$worksheet->set_row( 0, 20);
$worksheet->set_row( 1, undef, undef,1 );
$worksheet->set_row( 2, undef, undef, 0, 1 ,1);
```

#### set_column
> ( $first_col, $last_col, $width, $format, $hidden, $level, $collapsed )
```perl
$worksheet->set_column( 0, 0, 20 );    # Column  A   width set to 20
$worksheet->set_column( 1, 3, 30 );    # Columns B-D width set to 30
$worksheet->set_column( 'E:E', 20 );   # Column  E   width set to 20
$worksheet->set_column( 'F:H', 30 );   # Columns F-H width set to 30
```

#### set_default_row
>set_default_row( $height, $hide_unused_rows ) 
>隐藏所有未被使用的行

```perl
$worksheet->set_default_row( 24 ,1); 
```
#### set_tab_color
```perl
$worksheet2->set_tab_color( '#FF6600' );
```
#### freeze_panes
>freeze_panes( $y, $x, $top_row, $left_col )
```perl
$worksheet->freeze_panes( 1, 0 );    # Freeze the first row
$worksheet->freeze_panes( 'A2' );    # Same using A1 notation
$worksheet->freeze_panes( 0, 1 );    # Freeze the first column
$worksheet->freeze_panes( 'B1' );    # Same using A1 notation
$worksheet->freeze_panes( 1, 2 );    # Freeze first row and first 2 columns
$worksheet->freeze_panes( 'C2' );    # Same using A1 notation
```

#### hige_gridlines
>0 : Don't hide gridlines
1 : Hide printed gridlines only
2 : Hide screen and printed gridlines

## chart
```perl
$chart = $workbook->add_chart( type =>'column',embedded=>1 );
$worksheet->insert_chart( "A1" , $chart);
```

| type | graph |
| ---- | ---- |
| area |  面积图  |
| bar  | 横向柱状图  |
| column  |  纵向柱状图    |
| scatter| 分散图 |
|line| 线型图 |
|pie| 饼状图 |
|doughnut| 甜甜圈状图 |
|radar| 雷达图 |

### add_series
```perl
$chart->add_series(
    name       => 'chart1',
    categories => '=Sheet1!$A$2:$A$10', # Optional.
    values     => '=Sheet1!$B$2:$B$10', # Required.
    line       => { color => 'blue',  #  可以用 #FF0000 
                    width => 1.25,
                    dash_type => 'dash_dot',
                    transparency => 50,   #透明度
    },
    marker     => {
        type    => 'square',
        size    => 5,
        border  => { color => 'red' },
        fill    => { color => 'yellow' },
    },
);
```
### set_x_axis/set_y_axis
```perl
$chart->set_x_axis( name => "XADR",
　　　　　　　　　　　　name_font => { name => 'MsGothic', size => 10 } ,
　　　　　　　　　　　　max => 200,
                    min => 0,
                    major_unit => 20,
                    major_gridlines => { 
                        visible => 1,
                        line =>{ color => 'gray'},  			
                        transparency => 75,   
                    },
                    crossing => 0,
                    reverse => 0,
);

```

### set_title
```perl
$chart->set_title(
    name    => 'Title',
    overlay => 1,
    layout  => {
        x => 0.42,
        y => 0.14,
    }
);
```

### set_size
```perl
$chart->set_size( width  => 720,    # default 480
                  height => 576,    # default 288
                  # Same as
                  x_scale => 1.5,   # 倍数
                  y_scale => 2,
)
```
### set_legend
```perl
$chart->set_legend(
                   # none => 1,      #取消凡例
                   position => 'bottom', # 底部
)
```
### combine
```perl
$chart1->combine($chart2)
```
## format
```perl
$format = $workbook->add_format();
$format = $workbook->add_format(%font);

$format1 = $workbook->add_format();
$format1->copy($format);

%font = (
         font  => 'MsGothic',　
         size  => 12,
         color => 'blue',
         bold  => 1,
         bg_color => "yellow",
         top => 1,   # bottom left right
)
```

| mathod         | notice                       |
| :-------------- | :------------------------- |
| set_font()     | font                         |
| set_size()     | size                         |
| set_color()    | color                        |
| set_bold()     | bold                         |
| set_bg_color() | bg_color                     |
| set_align()    | align   # center  left right |
| set_top()      | top # bottom left right      |

