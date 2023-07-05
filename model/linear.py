


#张量并行


#TODO
#初始化参数
#模型参数设置
#机器划分
#GPU通行（一定要少的原则）
#矩阵切块
#其他模型
import torch 
import torch.nn as nn



class Linear(nn.Module):

	def __init__(self, input_size, output_size, gpu_list, bais=True):
		super().__init__()
		w, b = self.__get_init(input_size, output_size, gpu_list)
		#self.w = torch.nn.Parameter(w, requires_grad=True)
		#self.b = torch.nn.Parameter(b, requires_grad=True)
		self.gpu_list = gpu_list
		for i in range(len(gpu_list)):
			self.register_parameter("{}w".format(i), nn.Parameter(w[i], requires_grad=True))
			self.register_parameter("{}b".format(i), nn.Parameter(b[i], requires_grad=True))


	def __get_init(self, input_size, output_size, gpu_list):
		
		chunks = len(gpu_list)
		w = torch.empty(input_size, output_size)
		b = torch.empty(output_size)
		torch.nn.init.normal_(w, 0, 1)
		torch.nn.init.uniform_(b, 0, 0)
		return w.chunk(chunks, 1), b.chunk(chunks) 

	def single_gpu(self, intut_data, w, b, gpu):
		intut_data = input_data.cuda(gpu)
		w = w.cuda(gpu)
		b = b.cuda(gpu)
		return {gpu: torch.matmul(intut_data, w) + b}

	def forward(self, intut_data):
		a= []
		for i in range(len(self.gpu_list)):
			a.append(intut_data, eval("self.{}w".format(i)), eval("self.{}b".format(i)), self.gpu_list[i])
		a = [i.cuda(self.gpu_list[-1]) for i in a]
		return torch.cat(a, -1)



if __name__ == '__main__':
	n = Linear(1000, 1000, [0,1,2,3,4,5,6,7])
	a= torch.rand(10, 1000)
	b = n(a)
	print(b.shape)



