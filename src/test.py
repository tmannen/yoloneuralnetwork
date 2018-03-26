import numpy as np
S = 7
B = 2
C = 20

output = np.random.randn(S, S, C + B * 5)

bboxes = output[:,:,22:].reshape(S, S, B, 4)
confidences = output[:,:,20:20+B]
class_scores = output[:,:,:20]

max_scores = []
max_classes = []

for i in range(B):
	temp_scores = class_scores * confidences[:,:,i,None]
	max_scores.append(np.max(temp_scores, axis=2))
	max_classes.append(np.argmax(temp_scores, axis=2))

max_classes = np.stack(max_classes).transpose()
best_scores_per_cell = np.max(np.stack(max_scores), axis=0)
best_score_index = np.argmax(np.stack(max_scores), axis=0)

results = []

threshold = 0.95 * np.max(best_scores_per_cell)

for idxs in np.argwhere(best_scores_per_cell > threshold):
	row = idxs[0]
	col = idxs[1]
	result = {}
	bbox_idx = best_score_index[row,col]
	result['bbox'] = bboxes[row,col,bbox_idx,:]
	result['class'] = max_classes[row,col,bbox_idx]
	results.append(result)