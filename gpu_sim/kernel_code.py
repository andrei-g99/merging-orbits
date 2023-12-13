kernel_code = """

#define M_PI 3.141592653589793238462

struct vec3 {
    float x;
    float y;
    float z;
};

struct Body {
    float position[3];
    float velocity[3];
    float radius;
    float mass;
    int alive;
};

__device__ void subRoutine(Body *bodies, int i, int N, vec3 *accelOut) {
    
    float G = 0.1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < N && j < N)
    {

        //detect collisions
        if (i != j && Body[j].alive == 1)
        {
            vec3 distance;
            distance.x = Body[j].position[0] - Body[i].position[0];
            distance.y = Body[j].position[1] - Body[i].position[1];
            distance.z = Body[j].position[2] - Body[i].position[2];   
            float dist_norm = norm3df(distance.x, distance.y, distance.z);

            //detect collisions
            float radius_sum = m_j.radius + m_i.radius;
            if (dist_norm < radius_sum)
            {
                //collision has been detected between i and j bodies
                // The body with smaller mass is set to dead
                float m_i = Body[i].mass;
                float m_j = Body[j].mass;
                float total_mass = m_j + m_i;
                float new_radius = powf( total_mass * (3/(4*M_PI)), 1/3);

                
                vec3 velocity_of_merger;
                velocity_of_merger.x = (m_i / total_mass) * Body[i].velocity[0]
                                        + (m_j / total_mass) * Body[j].velocity[0];
                velocity_of_merger.y = (m_i / total_mass) * Body[i].velocity[1]
                                        + (m_j / total_mass) * Body[j].velocity[1];
                velocity_of_merger.z = (m_i / total_mass) * Body[i].velocity[2]
                                        + (m_j / total_mass) * Body[j].velocity[2];

                vec3 center_of_mass;
                center_of_mass.x = (m_i / total_mass) * Body[i].position[0]
                                        + (m_j / total_mass) * Body[j].position[0];
                center_of_mass.y = (m_i / total_mass) * Body[i].position[1]
                                        + (m_j / total_mass) * Body[j].position[1];
                center_of_mass.z = (m_i / total_mass) * Body[i].position[2]
                                        + (m_j / total_mass) * Body[j].position[2];

                if ( m_i <= m_j )
                {
                    Body[i].alive = 0;

                    Body[j].velocity[0] = velocity_of_merger.x;
                    Body[j].velocity[1] = velocity_of_merger.y;
                    Body[j].velocity[2] = velocity_of_merger.z;

                    Body[j].position[0] = center_of_mass.x;
                    Body[j].position[1] = center_of_mass.y;
                    Body[j].position[2] = center_of_mass.z;

                    Body[j].mass = total_mass;
                    Body[j].radius = new_radius;
                }
                else
                {
                    Body[j].alive = 0;

                    Body[i].velocity[0] = velocity_of_merger.x;
                    Body[i].velocity[1] = velocity_of_merger.y;
                    Body[i].velocity[2] = velocity_of_merger.z;

                    Body[i].position[0] = center_of_mass.x;
                    Body[i].position[1] = center_of_mass.y;
                    Body[i].position[2] = center_of_mass.z;

                    Body[i].mass = total_mass;
                    Body[i].radius = new_radius;
                }
                
                                        
            }


            // Compute forces
            vec3 acceleration_ij;
            float part = ((G * m_j) / (powf(dist_norm, 3)));
            acceleration_ij.x = distance.x * part;
            acceleration_ij.y = distance.y * part;
            acceleration_ij.z = distance.z * part;

            accelOut[j] = acceleration_ij;

        }


    }


}

__global__ void gravitySimulator(Body *bodies, Body *output, int N) {

    // X threads are total force calculations in parallel for each body
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    vec3 accelOut[N];

    if (i < N) {
        subRoutine<<<gridDim, blockDim>>>(bodies, i, N, accelOut);
    }
}

}
"""