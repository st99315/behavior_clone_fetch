import numpy as np

# # poly_info={'circle_color':circle_color, 'poly_type':dic['choice'], 'poly_color': dic[dic['choice']]['color']}
# load_poly_info=np.load("poly_info.npy")
# print(load_poly_info)
# print(type(load_poly_info))
# print(load_poly_info.item().get('circle_color'))
# print(load_poly_info.item().get('poly_type'))

a = 3
a_ary = np.array(a)
print('a_ary.shape = ' + str(a_ary.shape))

o = [3]
o_ary = np.array(o)
print('o_ary.shape = ' + str(o_ary.shape))

l = [3,4]
l_ary = np.array(l)
print('l_ary.shape = ' + str(l_ary.shape))
print(l_ary)

ll = [[3,4]]
ll_ary = np.array(ll)
print('ll_ary.shape = ' + str(ll_ary.shape))
print(ll_ary)
lll = [[[3,4]]]
lll_ary = np.array(lll)
print('lll_ary.shape = ' + str(lll_ary.shape))
print(lll_ary)
ll_squeeze = np.squeeze(ll_ary)
lll_squeeze = np.squeeze(lll_ary)
a_squeeze = np.squeeze(a_ary)
o_squeeze = np.squeeze(o_ary)



print('ll_squeeze.shape = ' + str(ll_squeeze.shape))
print(ll_squeeze)
print('lll_squeeze.shape = ' + str(lll_squeeze.shape))
print(lll_squeeze)

print('a_squeeze.shape = ' + str(a_squeeze.shape))
print(a_squeeze)
print('o_squeeze.shape = ' + str(o_squeeze.shape))
print(o_squeeze)

