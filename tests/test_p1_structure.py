import numpy as np
from neuronmimic.p1_structure import pair_split, pair_uniform, tree_features, fuse_blocks


def test_pair_split_direction_and_registry_order():
    ids=['720575940602553568','720575940602553569','720575940602553570']
    a=np.array([0,0,1]);b=np.array([1,2,2])
    assert np.array_equal(pair_split(ids,a,b),pair_split(ids,b,a))
    assert np.array_equal(pair_split(ids,a,b),pair_split(ids[::-1],2-a,2-b))
    assert len(np.unique(pair_uniform(ids,a,b)))==3


def test_heldout_pair_absent_in_training_including_reverse():
    ids=[str(720575940602553568+i) for i in range(100)]
    a,b=np.where(~np.eye(100,dtype=bool));s=pair_split(ids,a,b)
    train=set(zip(a[s==0],b[s==0])); held=set(zip(a[s!=0],b[s!=0]))
    assert not train&held
    assert not train&{(j,i) for i,j in held}
    assert set(s)=={0,1,2}


def test_contacts_and_partner_labels_cannot_change_geometry():
    d={'vertices':[[0,0,0],[1,0,0],[2,0,0]],'edges':[[0,1],[1,2]],
       'branch_per_edge':[0,0],'path_from_soma_nearest_um':[0,1,2]}
    x=tree_features(d)
    d.update(contacts=[{'target':'test','weight':999}],module='unseen-label')
    assert tree_features(d)==x
    assert x['cable_length_um']==2 and x['centroid_x_um']==1


def test_distance_enters_objective_with_declared_weight():
    g=np.zeros((2,4));s=np.array([[0,0,0],[3,4,0]]);t=np.zeros((2,2))
    x=fuse_blocks(g,s,t,2,0)
    assert np.isclose(np.sum((x[0]-x[1])**2),50)
    assert np.all(fuse_blocks(g,s,t,0,0)==0)
