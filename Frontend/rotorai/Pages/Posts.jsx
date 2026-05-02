'use client';
import { motion } from 'framer-motion';
import { useRouter, useSearchParams } from 'next/navigation';

const data = [
  //done
  {
    id: 1,
    title: 'RotorAI Concept & Problem Definition',
    content: '/sexy/1.png',
    text: 'RotorAI was initiated to address limitations in manual defect inspection.',
    text1:
      'The project began by defining the core problem scope, focusing on rust, cracks, and surface degradation. Early analysis highlighted challenges in traditional inspection methods, particularly in consistency and accessibility. A comparison between image-based and video-based approaches was conducted, revealing trade-offs in accuracy and scalability.',
    text2:
      'Key constraints such as lighting variation, occlusion, and dataset bias were identified early in development. One of the most significant early findings was the risk of false negatives — cases where real defects go undetected. In safety-critical industrial environments, a missed rust patch or crack can escalate into structural failure, making false negative minimisation a core design priority from the outset.',
    date: '15 SEPTEMBER 2025',
  },
  //
  {
    id: 2,
    title: 'Initial Rust Detection Prototype',
    content: '/sexy/2.png',
    text: 'Developed first prototype for rust detection using deep learning.',
    text1:
      'The first prototype implemented a basic deep learning model for rust detection. Image preprocessing techniques such as CLAHE were applied using OpenCV to enhance contrast and improve feature visibility. Initial testing was conducted on small-scale corrosion datasets, which demonstrated feasibility but also exposed limitations in generalisation across different environments.',
    text2:
      'A recurring issue at this stage was false negatives on low-contrast rust — areas where early-stage corrosion blended with the surrounding surface texture. The model failed to flag these regions, which highlighted the need for stronger preprocessing and more diverse training samples before moving to deployment.',
    date: '9 OCTOBER 2025',
  },
  //done
  {
    id: 3,
    title: 'RotorAI v1 – Image-Based Detection',
    content: '/sexy/3.png',
    text: 'Released first version with static image analysis pipeline.',
    text1:
      'RotorAI v1 introduced a TensorFlow-based segmentation pipeline using a U-Net architecture with a ResNet50 encoder. The system generated pixel-level masks for detecting rust and crack regions in high-resolution images. This phase established baseline performance metrics and demonstrated strong accuracy in controlled conditions.',
    text2:
      'However, testing revealed a pattern of false negatives at object edges and in shadowed regions. The model showed overconfidence in clean central surface areas, often missing defects near boundaries. This informed the need for augmentation strategies targeting edge cases and varied lighting conditions in future training rounds.',
    date: '30 OCTOBER 2025',
  },
  //done
  {
    id: 4,
    title: 'Dataset Expansion (v1.5)',
    content: '/sexy/4.png',
    text: 'Improved rust detection using larger datasets (> 40,000 Images ).',
    text1:
      "To improve detection performance, the DACL10k corrosion dataset was integrated into training. This significantly enhanced the model's ability to recognise varied rust patterns and textures. However, limitations became evident in crack detection, and the model showed signs of dataset bias due to over-reliance on corrosion-heavy samples.",
    text2:
      'The expanded dataset also introduced unexpected false negatives in crack detection — cracks that were thin, diagonal, or partially occluded were frequently missed. Additionally, the model began misclassifying dirt patches as rust due to colour similarity, a false positive pattern that would later require targeted negative sampling to resolve.',
    date: '16 NOVEMBER 2025',
  },
  //done
  {
    id: 5,
    title: 'Data Strategy Refinement',
    content: '/sexy/5.png',
    text: 'Shifted to multi-dataset approach for better generalisation.',
    text1:
      'A strategic shift was made to move away from reliance on a single dataset. Multiple Roboflow datasets were combined to increase diversity in defect types, environments, and lighting conditions. This approach improved generalisation and reduced class imbalance, resulting in more robust performance in real-world scenarios.',
    text2:
      'Critically, this phase introduced deliberate false negative mitigation. Samples where the model had previously failed — low-light rust, edge-adjacent cracks, partially occluded surfaces — were specifically targeted for inclusion. Retraining with these hard negatives pushed the model to attend to regions it had previously ignored, measurably reducing missed detections.',
    date: '25 NOVEMBER 2025',
  },
  //done
  {
    id: 6,
    title: 'RotorAI v2 – Dual Detection System',
    content: '../sexy/dual.png',
    text: 'Introduced combined rust and crack detection pipeline.',
    text1:
      'RotorAI v2 introduced a hybrid detection pipeline combining YOLOv8 for object detection with TensorFlow-based segmentation for refinement. This enabled multi-class detection of both rust and cracks within the same system. The updated pipeline improved detection accuracy and reduced false positives through better dataset diversity and model integration.',
    text2:
      'False negatives remained a concern in the video pipeline, particularly under motion blur and at low confidence thresholds. A confidence threshold of 0.65 was set as the operational baseline — below this, the rate of missed detections increased significantly. Frame skipping logic was also tuned to avoid skipping frames that captured unique defect viewpoints.',
    date: '10 DECEMBER 2025',
  },
  //done
  {
    id: 7,
    title: 'Real-Time Video Detection',
    content: '/sexy/sexy3.mp4',
    text: 'Enabled real-time detection using video streams.',
    text1:
      'The system was extended to support real-time video processing using OpenCV for frame extraction and YOLOv8 for inference. GPU acceleration was utilised to ensure efficient computation, enabling stable frame rates during live detection. This marked a key transition from static analysis to continuous inspection workflows.',
    text2:
      'One notable challenge was that camera quality directly influenced false negative rates. Lower resolution feeds produced blurrier frames, causing the model to miss defects it would otherwise detect clearly. To address this, users are advised to use high-quality cameras where possible, and the system documentation explicitly flags this as a known limitation affecting detection reliability.',
    date: '5 JANUARY 2026',
  },
  //
  {
    id: 8,
    title: '3D Gaussian Splatting Integration',
    content: './blogimg/Romans.png',
    text: 'Added 3D reconstruction for spatial visualisation.',
    text1:
      'To enhance interpretability, 3D Gaussian Splatting was integrated to reconstruct inspected environments from captured images. This allowed detected defects to be visualised within a spatial context, enabling users to better understand their location and distribution on complex surfaces.',
    text2:
      'An early approach of overlaying AI detection masks directly onto the 3D reconstruction was attempted but abandoned. Injecting binary masks into the reconstruction pipeline disrupted photometric consistency — a core requirement of 3DGS — causing geometry degradation and poor visual output. Keeping the detection and reconstruction pipelines separate proved essential to maintaining quality in both.',
    date: '20 FEBRUARY 2026',
  },
  //
  {
    id: 9,
    title: 'RaySplat Spatial Mapping',
    content: './blogimg/Thessalonians.jpg',
    text: 'Mapped 2D detections into 3D space using RaySplat.',
    text1:
      'RaySplat-based projection was implemented to map 2D detections into 3D space using ray-sphere intersection. This approach improved alignment on curved surfaces and allowed multiple detection views to be merged into a consistent spatial representation, significantly enhancing accuracy in 3D mapping.',
    text2:
      'The ray-sphere method was chosen over flat-plane or cylinder projections because the inspection images were captured across 270 degrees, making spherical projection the most geometrically accurate fit. For rust, detections from multiple viewpoints were merged into a single unified bounding volume to reduce visual clutter. For cracks, individual bounding boxes were preserved — merging them degraded spatial accuracy given their thin and distributed nature.',
    date: '13 MARCH 2026',
  },
  //done
  {
    id: 10,
    title: 'Final Report & System',
    content: '/sexy/40443486_ZeHueiLim.pdf',
    img2: '/sexy/sad.png',
    text: 'Final RotorAI system and dissertation submission.',
    text1:
      'The final system integrates image-based detection, real-time video processing, and 3D spatial visualisation into a unified pipeline. Performance was evaluated across multiple scenarios, and system limitations were analysed. The complete architecture, methodology, and results are documented in the final dissertation report.',
    text2:
      'A key takeaway from the project is the importance of false negative management in safety-critical applications. Throughout development, missed detections posed a greater risk than false positives — an undetected crack is far more dangerous than a flagged clean surface. Future work includes real-time 3D reconstruction, user authentication with detection history, and expanding the defect taxonomy beyond rust and cracks to support broader industrial inspection use cases.',
    date: '29 APRIL 2026',
    mygithub: 'https://github.com/ZEHUEI',
    myurl: 'https://www.linkedin.com/in/ze-huei-lim-310a162b5/',
    hisurl: 'https://www.linkedin.com/in/van-mien-7693b489/',
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

          <div className="mt-10">{post.text1}</div>

          {/* media (image / video / pdf) */}
          {post.id === 10 ? (
            <div className="mt-10 mb-10 flex flex-col items-center gap-6">
              {/* PDF Viewer */}
              <iframe
                src={post.content}
                className="w-full max-w-4xl h-[600px] rounded-lg shadow-lg"
              />

              <div>Links to our profiles</div>

              {/* Links */}
              <div className="flex flex-wrap gap-4 justify-center">
                {post.mygithub && (
                  <a
                    href={post.mygithub}
                    target="_blank"
                    className="border px-4 py-2 rounded-full hover:bg-white hover:text-black transition"
                  >
                    GitHub
                  </a>
                )}

                {post.myurl && (
                  <a
                    href={post.myurl}
                    target="_blank"
                    className="border px-4 py-2 rounded-full hover:bg-white hover:text-black transition"
                  >
                    My LinkedIn
                  </a>
                )}

                {post.hisurl && (
                  <a
                    href={post.hisurl}
                    target="_blank"
                    className="border px-4 py-2 rounded-full hover:bg-white hover:text-black transition"
                  >
                    Supervisor
                  </a>
                )}
              </div>
            </div>
          ) : post.content.endsWith('.mp4') ? (
            <video
              controls
              autoPlay
              loop
              muted
              playsInline
              className="mt-10 mb-10 rounded-lg max-w-3xl w-full mx-auto shadow-lg"
            >
              <source src={post.content} type="video/mp4" />
            </video>
          ) : (
            <img
              src={post.content}
              alt={post.title}
              className="mt-10 mb-10 rounded-lg max-w-3xl w-full mx-auto shadow-lg"
            />
          )}

          {post.text2 && <div className="mt-8 text-gray-400">{post.text2}</div>}
        </div>

        {post.id === 10 && post.img2 && (
          <div>
            <p className="mt-8 text-gray-400 font-AT">What I Am Doing Next?</p>
            <img
              src={post.img2}
              alt="Report Preview"
              className="mt-10 mb-10 rounded-lg max-w-3xl w-full mx-auto shadow-lg"
            />
          </div>
        )}
      </div>

      <div className="flex justify-center items-center mt-10 mb-10">
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
