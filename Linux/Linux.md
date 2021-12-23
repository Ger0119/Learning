# Linux
## 常用命令
### 操作
|命令|说明|
|:----|:----|
|ls [-adl] dir|列出目录 -a 全部文件 -d 目录本身 -l 长数据串|
|cd path|切换目录|
|pwd [-P]|显示当前所在目录 -P 真实目录|
|mkdir [-m] dir|创建目录 -m 配置权限 [mkdir -m 777 dir]|
|rmdir [-p] dir|删除目录 -p 归递删除|
|cp [-ipr] file dir|复制 -i 询问 -p 复制属性 -r 归递|
|touch file|创建文件|
|rm [-rif]|删除 -i 询问 -r 归递 -f 忽略警告|
|mv [-fi]|移动 -i 询问 -f 忽略警告|
### 查看
|命令|说明|
|:----|:----|
|cat [-bnTE] file|显示 -b(非空白行) -n 显示行号 -T tab -E $|
|tac file|从行尾显示内容|
|nl [-bnw] file|显示行号|
|more file|一页一页翻动 space下一页 Enter下一行 /str向下查找|
|less|一页一页翻动 [pageup]向上翻动 ?str向上查找 q离开|
|head [-n number] file|读取文件前n行|
|tail [-n number] file|读取文件后n行|
### 其他
1.grep
```shell
grep [-irv] str file/dir
# i 忽略大小写 r 递归查找 v 反向查抄
``` 
2.tar
```shell
tar [-zxcvf] file.tar.gz file
# z 调用gzip x 解压 c 打包 v显示过程 f 指定文件名
```
3.gzip
```shell
gzip [-dv] file 
# d 解压 v 显示过程
```

## 更改文件属性
### chmod 更改文件属性
每种身份(owner/group/others)各自的三个权限(r/w/x)是分数的累加 
> chmod [-R] nnn 文件名
> 
> r(read):4 w(write):2 x(execute):1 
```shell
chmod -R 777 data.log
chmod u=rwx,g=rx,o=r data.log 
# + 加入 - 除去 = 设定
```

### chown 更改文件属组/属主
> chown [-R] 属主名:属组名 文件名
> 
> -R 归递更改文件属性 更改文件夹内所有文件
```shell
chown -R root:root data.log
```

### chgrp 更改文件属组
> chgrp [-R] 属组名 文件名


