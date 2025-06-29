"""
Zaktualizowany skrypt do naprawy dodatku HybrIK dla Blendera 4.4+
Naprawia błędy:
1. use_auto_smooth (już naprawiony)
2. Brak obiektu Camera w scenie
"""
import os
import shutil
import re


def fix_blender_addon():
    """Fix HybrIK Blender addon compatibility issues with Blender 4.4+"""
    addon_path = r"C:\Users\xsiad\AppData\Roaming\Blender Foundation\Blender\4.4\scripts\addons\hybrik_blender_addon"
    convert2bvh_path = os.path.join(addon_path, "convert2bvh.py")

    if not os.path.exists(convert2bvh_path):
        print(f"Addon file not found: {convert2bvh_path}")
        return False

    print(f"Znaleziono plik: {convert2bvh_path}")

    # Utwórz backup tylko jeśli nie istnieje
    backup_path = convert2bvh_path + ".backup_v2"
    if not os.path.exists(backup_path):
        shutil.copy2(convert2bvh_path, backup_path)
        print(f"Utworzono backup: {backup_path}")

    # Read the file
    with open(convert2bvh_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix 1: Ensure use_auto_smooth is commented out
    if "ob.data.use_auto_smooth = False" in content and not "# ob.data.use_auto_smooth = False" in content:
        content = content.replace(
            "ob.data.use_auto_smooth = False",
            "# ob.data.use_auto_smooth = False  # autosmooth creates artifacts (disabled for Blender 4.4+)"
        )
        print("✅ Fix 1: Disabled use_auto_smooth")

    # Fix 2: Add safe camera access
    camera_pattern = r"cam_ob = bpy\.data\.objects\['Camera'\]"

    if re.search(camera_pattern, content):
        # Prepare the replacement code
        safe_camera_code = '''# Safe camera access - create camera if it doesn't exist
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        cam_ob = bpy.context.object
        cam_ob.name = 'Camera'
    else:
        cam_ob = bpy.data.objects['Camera']'''

        # Replace the problematic line
        content = re.sub(camera_pattern, safe_camera_code, content)
        print("✅ Fix 2: Added safe camera access with auto-creation")

    # Write the modified content back
    with open(convert2bvh_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n🎉 HybrIK Blender addon successfully patched!")
    print("Naprawione problemy:")
    print("   - use_auto_smooth (kompatybilność z Blender 4.4+)")
    print("   - Automatyczne tworzenie obiektu Camera jeśli nie istnieje")
    print("\nTeraz możesz spróbować ponownie zaimportować plik .pk w Blenderze")
    return True


if __name__ == "__main__":
    fix_blender_addon()
