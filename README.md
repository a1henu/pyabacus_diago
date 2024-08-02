# pyabacus_diago

编译无需添加其他参数：

```bash
cd diago
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

之后，在测试使用的python脚本中加入下面语句：

```python
import sys
sys.path.append('{PATH TO YOUR PYDIAGO REPOSITORY}/diago/build')
```

之后导入此module：

```python
import diago_dav_subspace
```

实例化一个 `Matrix_DiagDav`类的对象后调用 `diag`方法即可：

```python
matrix_diag = diago_dav_subspace.Matrix_DiagDav(
    h_mat,
    pre_condition,
    nband,
    nbasis,
    dav_ndim,
    diag_thr,
    diag_nmax,
    need_subspace,
    comm_info
)
res = matrix_diag.diag()
eigenvalue = matrix_diag.get_eigenvalue()
```
