# Win32::OLE

## 准备
```perl
use Cwd;
use Win32::OLE qw(in);
use Win32::OLE::Const 'Microsoft Excel';

$dir = getcwd;  #返回当前文件夹路径
$Excel = Win32::OLE->GetActiveObject('Excel.Application')||	
		 Win32::OLE->new('Excel.Application','Quit') or die;
$Excel->{DisplayAlerts}=0;  
```

## 打开Excel
```perl
$Book = $Excel->Workbooks->Open($dir."/book1.xlsx");  #需要绝对路径 #包括CSV文件
$Book = $Excel->Workbooks->Add;　#新建Excel
```
## Sheet 操作
```perl
$Book->Worksheets->Add->{Name} = "new";   #新建名为new的sheet
$sheet = $Book->Worksheets(1);            #从1开始 index指定sheet；
$sheet = $Book->Worksheets->Item('test'); #打开名为test的sheet
$sheet->{Name} = "newtest";               #更改sheet名
$Book->Worksheets('Sheet1')->Delete;      #删除sheet  输入Index也可以
$cnt = $Book->Worksheets->Count();        #返回sheet总数
foreach (in $Book->Worksheets){ print $_->{Name};}  #返回Sheet名
$sheet->Copy($sheet);                     #复制sheet到指定sheet前可跨文件
```
## 读取
```perl
$cell = $sheet->Cells(2,4)->{Value};  #row  column  相当与D2
$cell = $sheet->Range("D2")->{Value};
$data = $sheet->Range("A2:E4")->{Value};  #返回引用
$maxcol = $sheet->UsedRange->Find({What=>"*", 
                  SearchDirection=>xlPrevious,
                  SearchOrder=>xlByColumns})->{Column};
$maxrow = $sheet->UsedRange->Find({What=>"*",
    			  SearchDirection=>xlPrevious,
    		      SearchOrder=>xlByRows})->{Row};
    		      
foreach(@$data){	       #读取数据
    foreach(@$_){
        print $_."\t";
    }
    print "\n";
}
```
>11      12      13      14      15
21      22      23      24      25
31      32      33      34      35

## 写入
```perl
$sheet->Cells->{Value} = "A";
$sheet->Range("A8:C9")->{Value} = [[ undef, 'Xyzzy', 'Plugh' ],
                                   [ 42,    'Perl',  3.1415  ]];
#写入hyperlink
$worksheet->Hyperlinks->Add({ Anchor  => $worksheet->Range("A1:A1"),
                              Address => "Excel.xlsx#Sheet1!A5"});
@data = ();
push @data,[1,2,3,4,5];
push @data,[4,5,6,7,8];

$sheet->Range("A9:E10")->{Value} = \@data;  
#指定范围小于数据将只写入指定范围
#指定范围大于数据将被#N/A填充
$data->[0][0] = "A";       #[row][column] 
$data->[1][1] = "B";
$data->[0][2] = "C";
$sheet->Range("A1:C2")->{Value} = $data;
```
>A	C
         B

## 设置
```perl
$sheet->Range("A1:D1")->{AutoFilter} = 1;   #插入筛选filter
$sheet->Columns("A:C")->{ColumnWidth} = 10; #调整cell宽度
$sheet->Rows("1:3")->{RowHeight} = 20;      #调整cell高度
$sheet->Range("A1:D4")->Interior->{ColorIndex} = 27 #设置cell颜色 27是黄色max56
```
## 关闭Excel
```perl
$Book->Save;    #保存
$Book->SaveAs($dir."//other.xlsx");
$Book->Close;   #关闭
$Book->Quit;
```

