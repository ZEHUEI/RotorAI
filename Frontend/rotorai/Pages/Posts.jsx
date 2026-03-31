'use client';
import { motion } from 'framer-motion';
import { useRouter, useSearchParams } from 'next/navigation';

const data = [
  {
    id: 1,
    title: 'Introducing RotorAI: AI-Powered Motor Inspection',
    content: './blogimg/Blue.jpg',
    text: 'RotorAI was founded to solve a critical problem in industrial maintenance: defect detection in electric motors is still largely manual, time-consuming, and error-prone.',
    text1:
      'We designed  an end-to-end computer vision pipeline for motor inspectionn• Defined problem scope: corrosion, cracks, and surface defects\n• Evaluated feasibility of video-based detection vs static image models\n• Established initial dataset collection strategy using publicly available corrosion datasets\n• Identified challenges: lighting variance, surface reflectivity, and dataset imbalance',
    date: '2 OCOTBER 2025',
  },
  {
    id: 2,
    title: 'First Breakthrough: Rust Detection',
    content: './blogimg/Blue.jpg',
    text: 'Our first major milestone was achieving reliable rust detection.',
    text1:
      '• Trained YOLO model on early corrosion datasets\n• Used OpenCV to extract and preprocess frames from video streams\n• Applied data augmentation (rotation, brightness shifts) to improve robustness\n• Achieved consistent detection of rust patterns under controlled conditions\n• Identified limitations: poor generalisation on unseen industrial environments',
    date: '9 NOVEMBER 2025',
  },
  {
    id: 3,
    title: 'Launching V1 (MVP)',
    content: './blogimg/Blue.jpg',
    text: 'RotorAI V1 marks our Minimum Viable Product...',
    text1:
      '• Built full pipeline: video input → frame extraction → YOLO inference → output rendering\n• Integrated detection results into annotated video frames\n• Designed system to handle batch video uploads\n• Established baseline performance metrics for detection accuracy and latency\n• Highlighted need for more diverse training data to improve reliability',
    date: '10 DECEMBER 2025',
  },
  {
    id: 4,
    title: 'Scaling Accuracy (V1.5)',
    content: './blogimg/Blue.jpg',
    text: 'With V1.5, we focused on improving model accuracy...',
    text1:
      '• Trained models using the DACL10k dataset (industrial corrosion dataset)\n• Leveraged large-scale labelled data to improve detection consistency\n• Improved model performance on varied corrosion textures and densities\n• Observed dataset bias: strong on corrosion, weak on structural defects (cracks)\n• Identified overfitting risks due to dataset-specific characteristics',
    date: '20 DECEMBER 2025',
  },
  {
    id: 5,
    title: 'Pivoting Data Strategy',
    content: './blogimg/Blue.jpg',
    text: 'We pivoted from relying on a single dataset...',
    text1:
      '• Moved away from DACL10k due to limited defect diversity\n• Aggregated multiple Roboflow corrosion datasets for broader coverage\n• Combined datasets with different environments, lighting, and defect types\n• Improved generalisation across real-world conditions\n• Introduced dataset balancing to reduce class bias\n• Established a more scalable data pipeline for future model training',
    date: '25 DECEMBER 2025',
  },
  {
    id: 6,
    title: 'RotorAI V2: Multi-Defect Detection',
    content: './blogimg/Blue.jpg',
    text: 'RotorAI V2 introduces both corrosion and crack detection...',
    text1:
      '• Extended model to support multi-class detection (rust + cracks)\n• Integrated YOLO for detection with TensorFlow/PyTorch classifiers for refinement\n• Trained on combined corrosion and crack datasets from Roboflow\n• Improved detection accuracy across multiple defect categories\n• Optimised inference pipeline for near real-time performance\n• Reduced false positives through better dataset diversity',
    date: '16 JANUARY 2026',
  },

  {
    id: 7,
    title: 'Beyond Detection: Visualization',
    content: './blogimg/Blue.jpg',
    text: 'RotorAI is evolving beyond detection into visualization...',
    text1:
      '• Introduced 3D Gaussian Splatting for spatial visualization of defects\n• Mapped 2D detection outputs onto reconstructed 3D motor surfaces\n• Enabled spatial understanding of defect distribution\n• Designed pipeline for linking detection coordinates to 3D points\n• Focused on improving interpretability for engineers',
    date: 'MARCH',
  },
  {
    id: 8,
    title: 'CBA',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    text1: '',
    date: 'CBA',
    //     id: 8,
    // title: 'Final Product Vision',
    // image: './blogimg/Romans.png',
    // text: 'The final vision of RotorAI is a fully automated inspection platform...',
    // text1:
    //   '• Designed full-stack system: frontend dashboard + backend inference API\n• Built Flask-based API for handling model inference requests\n• Integrated real-time video processing capabilities\n• Planned scalable deployment using Google Cloud Run\n• Focused on delivering actionable insights rather than raw detections',
    // date: 'DD MONTH YEAR',
  },
  {
    id: 9,
    title: 'CBA',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    text1: '',
    date: 'CBA',
    // id: 9,
    // title: 'Validation & Deployment',
    // image: './blogimg/Thessalonians.jpg',
    // text: 'In the final phase, RotorAI is evaluated for real-world deployment...',
    // text1:
    //   '• Deployed backend using Google Cloud Run for scalability\n• Tested system under varying workloads and video input sizes\n• Evaluated latency vs accuracy trade-offs\n• Ensured API reliability for continuous inspection workflows\n• Designed system for integration into industrial environments',
    // date: 'DD MONTH YEAR',
  },
  {
    id: 10,
    title: 'RotorAI Impact',
    content: './blogimg/Blue.jpg',
    text: 'RotorAI demonstrates how AI can transform industrial inspection',
    text1: '',
    date: 'CBA',
    // text1:
    //   '• Delivered an AI-driven inspection system with real-world applicability\n• Reduced reliance on manual inspection processes\n• Enabled early detection of defects through automation\n• Demonstrated integration of AI, cloud, and 3D visualization\n• Showcased potential for predictive maintenance systems',
  },
];

const BlogIndividual = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const id = searchParams.get('id');
  const post = data.find((item) => item.id === Number(id));

  if (!post) return <div>Post not found</div>;
  return (
    <>
      <div className="w-full h-auto border-0 border-amber-300 rounded-lg p-2 text-gray-300">
        {/*change padding */}
        <div className="font-ocr mb-5 mt-5">[ {post.date} ]</div>
        <div className="font-AT text-4xl text-white mb-10">{post.title}</div>
        {/* <div className="text-base">{post.content}</div> */}
        <div className="font-Inter text-lg max-w-6xl leading-relaxed">
          {post.text}

          {/*have a picture here*/}
          <img src="placeholder" alt="placeholder" className="mt-15 mb-15" />

          <div className="mt-10">{post.text1}</div>
        </div>
      </div>

      <div className="flex justify-center items-center mt-10">
        <div
          className="border-2 rounded-full px-6 py-2 cursor-pointer transition-all hover:bg-white hover:text-black font-AT"
          onClick={() => router.push('/blog')}
        >
          BACK
        </div>
      </div>
    </>
  );
};

export default BlogIndividual;
