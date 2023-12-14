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
    float accel_due_to[{0}][3];
}};

__global__ void gravitySimulator(Body *bodies, Body *output, int N) {{

    float G = 0.1;
    int i = blockIdx.x * blockDim.x + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < N && j < N)
    {{
        float m_i = bodies[i].mass;
        float m_j = bodies[j].mass;

        //detect collisions
        if (i != j && bodies[j].alive == 1)
        {{
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
                    bodies[i].alive = 0;
                    bodies[i].velocity[0] = velocity_of_merger.x;
                    bodies[i].velocity[1] = velocity_of_merger.y;
                    bodies[i].velocity[2] = velocity_of_merger.z;
                    bodies[i].position[0] = center_of_mass.x;
                    bodies[i].position[1] = center_of_mass.y;
                    bodies[i].position[2] = center_of_mass.z;
                    bodies[i].mass = total_mass;
                    bodies[i].radius = new_radius;

                    bodies[j].alive = 1;
                    bodies[j].velocity[0] = velocity_of_merger.x;
                    bodies[j].velocity[1] = velocity_of_merger.y;
                    bodies[j].velocity[2] = velocity_of_merger.z;
                    bodies[j].position[0] = center_of_mass.x;
                    bodies[j].position[1] = center_of_mass.y;
                    bodies[j].position[2] = center_of_mass.z;
                    bodies[j].mass = total_mass;
                    bodies[j].radius = new_radius;
                }}
                else
                {{
                    bodies[j].alive = 0;
                    bodies[j].velocity[0] = velocity_of_merger.x;
                    bodies[j].velocity[1] = velocity_of_merger.y;
                    bodies[j].velocity[2] = velocity_of_merger.z;
                    bodies[j].position[0] = center_of_mass.x;
                    bodies[j].position[1] = center_of_mass.y;
                    bodies[j].position[2] = center_of_mass.z;
                    bodies[j].mass = total_mass;
                    bodies[j].radius = new_radius;

                    bodies[i].alive = 1;
                    bodies[i].velocity[0] = velocity_of_merger.x;
                    bodies[i].velocity[1] = velocity_of_merger.y;
                    bodies[i].velocity[2] = velocity_of_merger.z;
                    bodies[i].position[0] = center_of_mass.x;
                    bodies[i].position[1] = center_of_mass.y;
                    bodies[i].position[2] = center_of_mass.z;
                    bodies[i].mass = total_mass;
                    bodies[i].radius = new_radius;
                }}
                
                                        
            }}


            // Compute forces
            vec3 acceleration_ij;
            vec3 acceleration_ji;
            float part_i = ((G * m_j) / (powf(dist_norm, 3)));
            float part_j = ((G * m_i) / (powf(dist_norm, 3)));
            acceleration_ij.x = distance.x * part_i;
            acceleration_ij.y = distance.y * part_i;
            acceleration_ij.z = distance.z * part_i;

            acceleration_ji.x = (-distance.x) * part_j;
            acceleration_ji.y = (-distance.y) * part_j;
            acceleration_ji.z = (-distance.z) * part_j;

            if(i == j)
            {{
                bodies[i].accel_due_to[j][0] = 0;
                bodies[i].accel_due_to[j][1] = 0;
                bodies[i].accel_due_to[j][2] = 0;

                bodies[j].accel_due_to[i][0] = 0;
                bodies[j].accel_due_to[i][1] = 0;
                bodies[j].accel_due_to[i][2] = 0;
            }}
            else
            {{
            
                bodies[i].accel_due_to[j][0] = acceleration_ij.x;
                bodies[i].accel_due_to[j][1] = acceleration_ij.y;
                bodies[i].accel_due_to[j][2] = acceleration_ij.z;

                bodies[j].accel_due_to[i][0] = acceleration_ji.x;
                bodies[j].accel_due_to[i][1] = acceleration_ji.y;
                bodies[j].accel_due_to[i][2] = acceleration_ji.z;

            }}
        }}


        output[i] = bodies[i];
        output[j] = bodies[j];


    }}


}}
""".format(N)
