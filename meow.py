from torchft import Manager, ProcessGroupGloo
from datetime import timedelta
import time
from tqdm import tqdm

def load_state_dict(state_dict):
	#for j in tqdm(range(5), desc=f"Healing{i}"):
	#	time.sleep(1)
	print(f"Load state_dict")

def state_dict():
	return {}

pg = ProcessGroupGloo()
manager = Manager(
	pg=pg,
	min_replica_size=2,
	load_state_dict=load_state_dict,
	state_dict=state_dict,
	replica_id=f"train_ddp_test",
	timeout=timedelta(seconds=30),
	use_async_quorum=False,
)

while True:
	print("Pre start quorum")
	manager.start_quorum("start", allow_heal=True)
	print(f"Post start quorum with {manager.num_participants()} peers")
	for j in tqdm(range(5), desc=f"Inner steps {manager.current_step()}"):
		time.sleep(1)
	print("Pre ending quorum")
	manager.start_quorum("ending", allow_heal=False)
	print(f"Post ending quorum with {manager.num_participants()} peers")

	if manager.should_commit():
		print("Committed")
	if manager.current_step() > 20:
		break
