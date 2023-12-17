import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

N = config['simulation']['number_of_bodies'];


kernel_code = """

#define M_PI 3.141592653589793238462

struct vec3 {{
    float x;
    float y;
    float z;
}};

struct Body {{
    float position[3];
    float velocity[3];
    float radius;
    float mass;
    int alive;
}};


__global__ void gravitySimulator(Body *bodies, Body *output, int* alive_cnt, int N, float G, float dt) {{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < N )
    {{
    
        float total_accel_x = 0;
        float total_accel_y = 0;
        float total_accel_z = 0;
        
        output[i] = bodies[i];
        
        for(int j = 0; j < N; j++)
        {{

            //detect collisions
            if (i != j && bodies[j].alive == 1)
            {{
                float m_i = bodies[i].mass;
                float m_j = bodies[j].mass;
                vec3 distance;
                distance.x = bodies[j].position[0] - bodies[i].position[0];
                distance.y = bodies[j].position[1] - bodies[i].position[1];
                distance.z = bodies[j].position[2] - bodies[i].position[2];   
                float dist_norm = norm3df(distance.x, distance.y, distance.z);

                //detect collisions
                float radius_sum = bodies[j].radius + bodies[i].radius;
                if (dist_norm < radius_sum)
                {{
                    //collision has been detected between i and j bodies
                    // The body with smaller mass is set to dead
                    float total_mass = m_j + m_i;
                    float new_radius = powf( total_mass * (3/(4*M_PI)), 1/3);

                    
                    vec3 velocity_of_merger;
                    velocity_of_merger.x = (m_i / total_mass) * bodies[i].velocity[0] + (m_j / total_mass) * bodies[j].velocity[0];
                    velocity_of_merger.y = (m_i / total_mass) * bodies[i].velocity[1] + (m_j / total_mass) * bodies[j].velocity[1];
                    velocity_of_merger.z = (m_i / total_mass) * bodies[i].velocity[2] + (m_j / total_mass) * bodies[j].velocity[2];

                    vec3 center_of_mass;
                    center_of_mass.x = (m_i / total_mass) * bodies[i].position[0] + (m_j / total_mass) * bodies[j].position[0];
                    center_of_mass.y = (m_i / total_mass) * bodies[i].position[1] + (m_j / total_mass) * bodies[j].position[1];
                    center_of_mass.z = (m_i / total_mass) * bodies[i].position[2] + (m_j / total_mass) * bodies[j].position[2];

                    if ( m_i <= m_j )
                    {{
                        output[i].alive = 0;
                        atomicSub(alive_cnt, 1);
                        output[i].velocity[0] = velocity_of_merger.x;
                        output[i].velocity[1] = velocity_of_merger.y;
                        output[i].velocity[2] = velocity_of_merger.z;
                        output[i].position[0] = center_of_mass.x;
                        output[i].position[1] = center_of_mass.y;
                        output[i].position[2] = center_of_mass.z;
                        output[i].mass = total_mass;
                        output[i].radius = new_radius;

                        output[j].alive = 1;
                        output[j].velocity[0] = velocity_of_merger.x;
                        output[j].velocity[1] = velocity_of_merger.y;
                        output[j].velocity[2] = velocity_of_merger.z;
                        output[j].position[0] = center_of_mass.x;
                        output[j].position[1] = center_of_mass.y;
                        output[j].position[2] = center_of_mass.z;
                        output[j].mass = total_mass;
                        output[j].radius = new_radius;
                    }}
                    else
                    {{
                        output[j].alive = 0;
                        atomicSub(alive_cnt, 1);
                        output[j].velocity[0] = velocity_of_merger.x;
                        output[j].velocity[1] = velocity_of_merger.y;
                        output[j].velocity[2] = velocity_of_merger.z;
                        output[j].position[0] = center_of_mass.x;
                        output[j].position[1] = center_of_mass.y;
                        output[j].position[2] = center_of_mass.z;
                        output[j].mass = total_mass;
                        output[j].radius = new_radius;

                        output[i].alive = 1;
                        output[i].velocity[0] = velocity_of_merger.x;
                        output[i].velocity[1] = velocity_of_merger.y;
                        output[i].velocity[2] = velocity_of_merger.z;
                        output[i].position[0] = center_of_mass.x;
                        output[i].position[1] = center_of_mass.y;
                        output[i].position[2] = center_of_mass.z;
                        output[i].mass = total_mass;
                        output[i].radius = new_radius;
                    }}
                    
                                            
                }}


                // Compute forces
                vec3 acceleration_ij;
                vec3 acceleration_ji;
                float part_i = ((G * m_j) / (powf(dist_norm, 3)));
                acceleration_ij.x = distance.x * part_i;
                acceleration_ij.y = distance.y * part_i;
                acceleration_ij.z = distance.z * part_i;

                total_accel_x += acceleration_ij.x;
                total_accel_y += acceleration_ij.y;
                total_accel_z += acceleration_ij.z;

            }}


        }}

        //add contributions from each body
        output[i].velocity[0] = output[i].velocity[0] + (total_accel_x * dt);
        output[i].velocity[1] = output[i].velocity[1] + (total_accel_y * dt);
        output[i].velocity[2] = output[i].velocity[2] + (total_accel_z * dt);

        output[i].position[0] = output[i].position[0] + (output[i].velocity[0] * dt);
        output[i].position[1] = output[i].position[1] + (output[i].velocity[1] * dt);
        output[i].position[2] = output[i].position[2] + (output[i].velocity[2] * dt);


    }}



}}
""".format(N)
