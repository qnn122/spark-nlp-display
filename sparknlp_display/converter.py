
from sparknlp_display import RelationExtractionVisualizer
from sparknlp.annotation import Annotation
from typing import Dict

def pkre2sparknlp(pkre: Dict):
	"""
	Arguments:
		pkre (dict): prediction from pk relation model
			'text': text body
			'relations': relation detected
	
	Return:
		(dict): structure resembling results of sparknlp pipeline
	"""
	# Initialize output
	text = pkre['text']
	result = {
		'document': [Annotation('document', 0, len(text)-1, text, (), None)],
		'relations': []
	}

	# converting
	res = pkre['relations'] # relations (list)
	for re in res:
		meta = {
			'entity2': re['head_span']['label'],
			'entity2_begin': re['head_span']['start'],
			'entity2_end': re['head_span']['end'],
			'chunk2': text[re['head_span']['start']:re['head_span']['end']],

			'entity1': re['child_span']['label'],
			'entity1_begin': re['child_span']['start'],
			'entity1_end': re['child_span']['end'],
			'chunk1': text[re['child_span']['start']:re['child_span']['end']]
		}

		result['relations'].append(Annotation(annotator_type=None,
											begin=None,
											end=None,
											result=re['label'],
											metadata=meta,
											embeddings=None))
	return result