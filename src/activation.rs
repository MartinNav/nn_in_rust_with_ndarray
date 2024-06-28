
pub struct Activation {
    pub function: fn(&f32) -> f32,
    pub derivation: fn(&f32) -> f32,
}
impl Activation {
    /// This function will instantiate softmax activation function
    pub fn softmax()->Self{
Activation {
        function: |x: &f32| -> f32 {
            if *x > 0. {
                return *x;
            } else {
                0.
            }
        },
        derivation: |x: &f32| -> f32 {
            if *x > 0. {
                return 1.0;
            } else {
                0.
            }
        },
    }

    }


    /// x^3+1+(sin 9x)*0.1 function
    pub fn experimental_x3()->Self{
        let f = |x:&f32|->f32{
            x.powi(3)+1.+f32::sin(9.*x)*0.1
        };
        let d = |x:&f32|->f32{
            3.*x.powi(2)+f32::cos(9.*x)
        };
        Activation{
            function:f,
            derivation:d,
        }
    }
    
}
