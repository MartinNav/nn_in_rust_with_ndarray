use ndarray::prelude::*;
use rand::Rng;
mod activation;
use crate::activation::Activation;
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Array2<f32>>,
    bias: Vec<Array2<f32>>,
    data: Vec<Array2<f32>>,
    activation: Activation,
    learning_rate: f32,
}
impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut w = vec![];
        let mut b = vec![];
        for i in 0..layers.len() - 1 {
            let mut wm = Array::zeros([layers[i + 1], layers[i]]);
            let mut bm = Array::zeros([layers[i + 1], 1]);

            let _ = wm
                .iter_mut()
                .map(|m| *m += rng.gen::<f32>())
                .collect::<Vec<_>>();
            let _ = bm
                .iter_mut()
                .map(|m| *m += rng.gen::<f32>())
                .collect::<Vec<_>>();
            w.push(wm);
            b.push(bm);
        }
        Network {
            layers: layers,
            weights: w,
            bias: b,
            data: vec![],
            activation: activation,
            learning_rate: learning_rate,
        }
    }
    pub fn print(&self) {
        println!("weights:");
        for w in &self.weights {
            println!("{w:?}");
        }
        for b in &self.bias {
            println!("{b:?}");
        }
    }
    pub fn feed_forward(&mut self, inputs: Array2<f32>) -> Array2<f32> {
        let mut current = inputs;
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            current = &self.weights[i].dot(&current) + &self.bias[i];
            let _: Vec<_> = current
                .iter_mut()
                .map(|m| *m = (self.activation.function)(m))
                .collect();
            self.data.push(current.clone());
        }
        current
    }
    pub fn back_prop(&mut self, inputs: Array2<f32>, targets: Array2<f32>) {
        let mut errror = &targets - &inputs;
        let mut gradients = inputs.map(self.activation.derivation);
        for i in (0..self.layers.len() - 1).rev() {
            gradients = (&gradients * &errror).map(|x| x * self.learning_rate);

            self.weights[i] = &self.weights[i] + (&gradients.dot(&self.data[i].t()));
            self.bias[i] = &self.bias[i] + (&gradients);

            errror = self.weights[i].t().dot(&errror);
            gradients = self.data[i].map(self.activation.derivation);
        }
    }
    pub fn train(&mut self, inputs: Vec<Array2<f32>>, targets: Vec<Array2<f32>>, epochs: u32) {
        for i in 1..=epochs {
            if (i%100)==0 {
                println!("Epoch {i} out of {epochs}");
            }
            for j in 0..inputs.len() {
                let out = self.feed_forward(inputs[j].clone());
                self.back_prop(out, targets[j].clone());
            }
        }
    }
}
fn main() {
    //let mut nn= Network::new(vec![2, 4, 4, 1], activation, 0.01);
    let mut nn = Network::new(vec![2, 4, 4, 1], Activation::softmax(), 0.01);
    println!("NOW TRAINING:");
    //nn.print();
    let inputs = vec![
        Array2::from(vec![[0.0], [0.0]]),
        Array2::from(vec![[1.0], [0.0]]),
        Array2::from(vec![[0.0], [1.0]]),
        Array2::from(vec![[1.0], [1.0]]),
    ];
    let outputs = vec![
        Array2::from(vec![[0.0]]),
        Array2::from(vec![[0.0]]),
        Array2::from(vec![[0.0]]),
        Array2::from(vec![[1.0]]),
    ];
    println!(
        "before{:?}",
        nn.feed_forward(Array2::from(vec![[0.0], [1.0]]))
    );
    nn.train(inputs, outputs, 3000);
    println!(
        "after{:?}",
        nn.feed_forward(Array2::from(vec![[0.0], [1.0]]))
    );
    println!("AFTER TRAINING:");
    //nn.print();
    println!(
        "1 1 {}",
        nn.feed_forward(Array2::from(vec![[1.0], [1.0]]))
            .into_raw_vec()[0]
    );
    println!(
        "0 1 {}",
        nn.feed_forward(Array2::from(vec![[0.0], [1.0]]))
            .into_raw_vec()[0]
    );
    println!(
        "1 0 {}",
        nn.feed_forward(Array2::from(vec![[1.0], [0.0]]))
            .into_raw_vec()[0]
    );
    println!(
        "0 0 {}",
        nn.feed_forward(Array2::from(vec![[0.0], [0.0]]))
            .into_raw_vec()[0]
    );
}
