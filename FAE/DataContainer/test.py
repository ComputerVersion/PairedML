import numpy as np
import pandas as pd
#concatenating

df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
# df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
# res = pd.concat([df1,df2,df3], axis=0,ignore_index=True)

# res = pd.concat([df2,df1],join='inner',ignore_index=True)
#join ,['inner','outer']

res = pd.concat([df1,df2],axis=1,join_axes = [df1.index])

print(res)






