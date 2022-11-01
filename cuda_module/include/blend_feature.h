void launch_add2(float *c,
                 const float *nodes,
                 const float *pts,
                 const float r,
                 int n_nodes);
//given posed nodes & sample_d pts 
//every nodes has its‘ influence range r
//first find the number of pts in side nodes_i's field , marked as si
//then find the maximum s' = max(s1, s2, si)