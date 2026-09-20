# Git Bash 快速入门指南

> 本文档面向对 Git Bash（即在 Windows 上使用的 Bash 环境）不熟悉的同学，提供**基础概念**、**常用命令**、**脚本模板**以及**练习示例**，帮助你快速上手并在项目中自如使用。

---

## 1. 什么是 Git Bash？
- **Git Bash** 是 Windows 上的一个轻量级 **类 Unix** shell，随 Git for Windows 一起提供。
- 它提供了 Bash (Bourne Again SHell) 语法、常见的 GNU 工具（`ls`, `grep`, `sed` 等），并且已经预装了 `git` 命令。
- 适合在 Windows 环境下完成 **脚本编写、版本控制、文件操作** 等任务。

---

## 2. 常用概念与语法
| 概念 | 示例 | 说明 |
|------|------|------|
| **变量** | `NAME="ZJY"` | 使用 `=` 赋值，字符串可加引号，引用时前加 `$`。
| **环境变量** | `export PATH="$PATH:/my/tool"` | `export` 让变量在子进程中可见。
| **数组** | `files=(a.txt b.txt c.txt)` | 通过索引访问 `${files[0]}`。
| **条件判断** | `if [ -f file.txt ]; then echo "exists"; fi` | `[` 与 `]` 是 `test` 命令的别名，常用于文件/字符串检测。
| **循环** | `for f in *.py; do echo $f; done` | Bash 支持 `for`, `while`, `until` 循环。
| **函数** | `my_func(){ echo "Hello $1"; }` | 定义后可直接调用 `my_func World`。
| **脚本执行** | `chmod +x script.sh && ./script.sh` | 需要给文件可执行权限。

---

## 3. 常用 Git Bash 命令速查表
| 类别 | 命令 | 用途 |
|------|------|------|
| **文件/目录** | `ls`, `cd`, `mkdir`, `rm`, `cp`, `mv` | 浏览、切换、创建、删除、复制、移动。
| **文本处理** | `cat`, `grep`, `awk`, `sed`, `cut`, `sort`, `uniq` | 查看、搜索、过滤、替换、列切分、排序、去重。
| **管道 & 重定向** | `|`, `>`, `>>`, `<` | 把前一个命令的输出作为后一个命令的输入；输出重定向到文件。
| **压缩** | `tar -czf archive.tar.gz dir/`, `unzip file.zip` | 打包/解压。
| **系统信息** | `pwd`, `whoami`, `date`, `uptime` | 打印当前目录、用户名、时间、系统运行时长。
| **网络** | `ssh`, `scp`, `wget`, `curl` | 远程登录/文件拷贝/下载。
| **Git** | `git status`, `git add`, `git commit -m "msg"`, `git push` | 基础版本控制操作。

---

## 4. 常用脚本模板
### 4.1 参数解析模板（使用 `getopts`）
```bash
#!/usr/bin/env bash

# 用法示例: ./run.sh -e dev -p 8080 -v

while getopts ":e:p:vh" opt; do
  case $opt in
    e) ENV=$OPTARG;;           # 环境，如 dev / prod
    p) PORT=$OPTARG;;          # 端口号
    v) VERBOSE=1;;             # 开启详细模式
    h) echo "Usage: $0 -e <env> -p <port> [-v]"; exit 0;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

echo "Running in env: $ENV on port: $PORT"
[[ $VERBOSE ]] && set -x   # 如果开启详细模式，打印每一步命令
# 在此写业务逻辑
```

### 4.2 批量处理文件的模板
```bash
#!/usr/bin/env bash
# 将 target_dir 下所有 .txt 文件压缩为 .gz 并移动到 archive_dir

target_dir="/data/input"
archive_dir="/data/archive"

mkdir -p "$archive_dir"

for f in "$target_dir"/*.txt; do
  [ -e "$f" ] || continue   # 若没有匹配文件则跳过
  gzip -c "$f" > "${f}.gz"
  mv "${f}.gz" "$archive_dir/"
  echo "Archived $f"
 done
```

### 4.3 远程执行并同步结果（常用于实验室服务器）
```bash
#!/usr/bin/env bash
# 将本地脚本上传并在远程执行，随后把结果拉回本地
REMOTE="zhoujiayan@10.130.136.134"
PORT=15633
KEY="~/.ssh/id_ed25519"

# 1. 上传脚本
scp -P $PORT -i $KEY train.sh $REMOTE:/tmp/
# 2. 远程执行（nohup 防止断连）
ssh -p $PORT -i $KEY $REMOTE "nohup bash /tmp/train.sh > /tmp/train.log 2>&1 &"
# 3. 等待数秒后拉回日志
sleep 5
scp -P $PORT -i $KEY $REMOTE:/tmp/train.log ./train.log
```

---

## 5. 练习：写一个简单的文件统计脚本
**目标**：遍历当前目录的所有子目录，统计每个子目录中 `.py` 文件的行数，并将结果保存为 `summary.txt`。

```bash
#!/usr/bin/env bash
# stats.sh

output="summary.txt"
> "$output"   # 清空旧文件

for d in */; do
  total=0
  for f in "$d"*.py; do
    [ -e "$f" ] || continue
    lines=$(wc -l < "$f")
    total=$((total + lines))
  done
  echo "${d%/}: $total lines" >> "$output"
 done

cat "$output"
```

运行方式：
```bash
chmod +x stats.sh
./stats.sh
```

---

## 6. 常见坑 & 调试技巧
- **脚本报 `command not found`**：检查文件是否有可执行权限 (`chmod +x script.sh`)；或者在第一行加入 `#!/usr/bin/env bash`。
- **变量未展开**：在双引号中使用变量 (`"$var"`)；单引号会原样输出。
- **路径分隔符**：在 Bash 中使用正斜杠 `/`，即使在 Windows 上也一样。
- **调试**：在脚本顶部加 `set -euo pipefail`（错误即退出、未定义变量报错、管道错误传播），或者在需要查看执行过程时加 `set -x`。

---

## 7. 进一步学习资源
- 官方文档: <https://www.gnu.org/software/bash/manual/bash.html>
- 《Learning the Bash Shell》（O'Reilly）
- 《Advanced Bash-Scripting Guide》 (免费在线) <http://tldp.org/LDP/abs/html/>
- Git 官方手册: <https://git-scm.com/doc>

祝你在 Windows + Git Bash 环境中玩得开心 🚀！
