from data.tools import load_nc
smb_data = load_nc("./dataset/CSR_grid_DDK3.nc")
print(smb_data.max())