import tifffile

name = r'H:\血管后处理文献阅读\新增参考文献\graph\tutorials\3d_classification\datasets\dataset40\train\0000-0069\0000-0069_3dim.tif'
f = tifffile.imread(name)
print(f.shape)

tifffile.imwrite(r'H:\血管后处理文献阅读\新增参考文献\graph\tutorials\3d_clas'
                 r'sification\datasets\dataset40\train\0000-0069\0.tif', f[0])
tifffile.imwrite(r'H:\血管后处理文献阅读\新增参考文献\graph\tutorials\3d_clas'
                 r'sification\datasets\dataset40\train\0000-0069\1.tif', f[1])
tifffile.imwrite(r'H:\血管后处理文献阅读\新增参考文献\graph\tutorials\3d_clas'
                 r'sification\datasets\dataset40\train\0000-0069\2.tif', f[2])