import numpy as np
import cv2


def points_sampling(mask, num=4):
    mask = np.array(mask)

    # mask = mask.reshape((-1, 1, 2))
    # mask = cv2.convexHull(mask, returnPoints=True)
    # mask = mask.reshape((-1, 2))

    pnum, cnum = mask.shape
    assert cnum == 2
    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = mask[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - mask) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > num:
        edgeidxkeep_k = edgeidxsort_p[pnum - num:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = mask[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == num
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * num / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != num:

            if edgenumsum > num:

                id = -1
                passnum = edgenumsum - num
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += num - edgenumsum

        assert np.sum(edgenum) == num

        psample = []
        for i in range(pnum):
            pb_1x2 = mask[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def mask_sampling(masks, num=8):
    return np.array([points_sampling(m, num) for m in masks], np.float32)

if __name__ == '__main__':
    mask1 = np.array([[[10, 0], [5, 10], [0, 10], [10, 5], [0, 0], [5, 0], [10, 10], [0, 5]],
                      [[0, 0], [5, 0], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]]])
    mask2 = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]])

    points = mask_sampling(mask2, num=16)
    print(points)
