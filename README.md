<div style="display: flex; flex-direction: row;">
<img src='assets/logo_square.png' style="height:130px;border-radius: 25px;margin: 10px;"></img>
<h1 style="margin-top:40px;">StereoCrafter-Zero: Zero-Shot Stereo Video Generation with Noisy Restart</h1>
</div>
<div align="center">

<br><br>

<img src='assets/banner.png' style="width:90%"></img>

</div>

 
## 🔆 Introduction
🔥🔥 Our method performs stereo video generation with only a single image and a prompt as input. We show anaglyph for best showing the stereoscopic effects.
<br>


### 1.1. Showcases (576x1024)
<table class="center">
  <!-- <tr>
    <td colspan="1">"fireworks display"</td>
    <td colspan="1">"a robot is walking through a destroyed city"</td>
  </tr> -->
  <tr>
  <td>
    <img src=assets/showcase/1024/fireworks_display-r.gif height="200">
  </td>
  <td>
    <img src=assets/showcase/1024/fireworks_display-r_overlay.gif height="200">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/showcase/1024/a_beautiful_woman_with_long_hair_and_a_d-r.gif  height="200">
  </td>
  <td>
    <img src=assets/showcase/1024/a_beautiful_woman_with_long_hair_and_a_d-r_overlay.gif  height="200">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/showcase/1024/a_robot_is_walking_through_a_destroyed_c-r.gif  height="200">
  </td>
  <td>
    <img src=assets/showcase/1024/a_robot_is_walking_through_a_destroyed_c-r_overlay.gif  height="200">
  </td>
  </tr>
</table>


### 1.2. Showcases (320x512)
<table class="center">
  <tr>
  <td>
    <img src=assets/showcase/512/a_bonfire_is_lit_in_the_middle_of_a_fiel-r.gif height="200">
  </td>
  <td>
    <img src=assets/showcase/512/a_bonfire_is_lit_in_the_middle_of_a_fiel-r_overlay.gif height="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/showcase/512/a_sailboat_sailing_in_rough_seas_with_a_-r.gif height="200">
  </td>
  <td>
    <img src=assets/showcase/512/a_sailboat_sailing_in_rough_seas_with_a_-r_overlay.gif height="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/showcase/512/a_woman_looking_out_in_the_rain-r.gif height="200">
  </td>
  <td>
    <img src=assets/showcase/512/a_woman_looking_out_in_the_rain-r_overlay.gif height="200">
  </td>
  </tr>
</table>




### 1.3. Showcases (256x256)


<table class="center">
  <tr>
  <td>
    <img src=assets/showcase/256/a_campfire_on_the_beach_and_the_ocean_wa-r.gif height="180">
  </td>
  <td>
    <img src=assets/showcase/256/a_campfire_on_the_beach_and_the_ocean_wa-r_overlay.gif height="180">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/showcase/256/bear_playing_guitar_happily,_snowing-r.gif height="180">
  </td>
  <td>
    <img src=assets/showcase/256/bear_playing_guitar_happily,_snowing-r_overlay.gif height="180">
  </td>
  </tr>
</table>

### 2. Applications

#### 2.2 Generative frame interpolation

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Generated stereo video</td>
    </tr>
   <tr>
  <td>
    <img src=assets/application/smile_start.png height="120">
  </td>
  <td>
    <img src=assets/application/smile_end.png height="120">
  </td>
  <td>
    <div style="display: flex">
      <img src=assets/showcase/interp/a_smiling_girl-r.gif height="120">
      <img src=assets/showcase/interp/a_smiling_girl-r_overlay.gif height="120">
    </div>
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/application/stone01_start.png height="120">
  </td>
  <td>
    <img src=assets/application/stone01_end.png height="120">
  </td>
  <td>
    <div style="display: flex">
      <img src=assets/showcase/interp/rotating_view-r.gif height="120">
      <img src=assets/showcase/interp/rotating_view-r_overlay.gif height="120">
    </div>
  </td>
  </tr> 
</table >

#### 2.3 Looping video generation
<table class="center">

  <tr>
  <td>
    <img src=assets/showcase/interp/clothes_swaying_in_the_wind-r.gif height="150">
  </td>
  <td>
    <img src=assets/showcase/interp/clothes_swaying_in_the_wind-r_overlay.gif height="150">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/showcase/interp/flowers_swaying_in_the_wind-r.gif height="150">
  </td>
  <td>
    <img src=assets/showcase/interp/flowers_swaying_in_the_wind-r_overlay.gif height="150">
  </td>
  </tr>


</table >
